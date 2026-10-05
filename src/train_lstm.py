"""
Training of the LSTM learners for the LTMLE nuisance regressions (rebuilt 2026-10).

  fit_treatment(arm)   g_t(a | history), t = 0..K, one many-to-many model per replicate:
                       arm = "multinomial" -> one softmax model over the J treatments;
                       arm = "binomial"    -> J separate sigmoid models (one-vs-rest), not renormalised.
  fit_censoring()      P(C_t = 0 | history, A_t), t = 1..K, one many-to-many sigmoid model.
  fit_outcome_step()   one iterated-conditional-expectation (ICE) regression at step s:
                       pseudo-outcome Z_s in [0, 1] on the last HP["outcome_window"] months of history
                       ending at s (soft-label binary cross-entropy), warm-started from the previous
                       step of the same recursion.

Losses are masked to the relevant risk sets with per-time sample weights; no class weighting is
used, so predicted probabilities stay calibrated. Early stopping uses a subject-level validation split.
"""

import numpy as np

import utils
from utils import HP, STATE
import test_lstm


class _BestWeights:
    """Keep the weights of the epoch with the lowest validation loss and restore them at the end."""

    def __new__(cls, patience):
        keras = STATE["keras"]

        class BestWeights(keras.callbacks.Callback):
            def __init__(self, patience):
                super().__init__()
                self.patience = int(patience)
                self.best = np.inf
                self.best_weights = None
                self.wait = 0
                self.epochs_run = 0

            def on_epoch_end(self, epoch, logs=None):
                self.epochs_run = epoch + 1
                cur = (logs or {}).get("val_loss", (logs or {}).get("loss"))
                if cur is not None and np.isfinite(cur) and cur < self.best - 1e-6:
                    self.best, self.wait = cur, 0
                    self.best_weights = self.model.get_weights()
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.model.stop_training = True

            def on_train_end(self, logs=None):
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)

        return BestWeights(patience)


def _fit(model, x, y, w, rows, seed, max_epochs, patience):
    tr, va = utils.split_rows(rows, HP["val_fraction"], seed)
    cb = _BestWeights(patience)
    val = (x[va], y[va], w[va]) if len(va) else None
    model.fit(x[tr], y[tr], sample_weight=w[tr], validation_data=val, epochs=int(max_epochs),
              batch_size=int(HP["batch_size"]), callbacks=[cb], verbose=0, shuffle=True)
    return cb.epochs_run


def _treatment_inputs():
    d = STATE["data"]
    return np.concatenate([d["base"], utils.lagged(d["A_onehot"])], axis=-1)


def _censoring_inputs():
    d = STATE["data"]
    base_no_age = np.delete(d["base"], 5, axis=-1)  # column 5 is V3 (age); see fit_censoring_models()
    return np.concatenate([base_no_age, utils.lagged(d["A_onehot"]), d["A_onehot"]], axis=-1)


def fit_treatment(arm="multinomial"):
    """Returns (probs (n, T, J), epochs) with probabilities at every subject and time."""
    d = STATE["data"]
    J = d["J"]
    x = _treatment_inputs()
    w = d["at_risk"]
    rows = np.where(w.sum(axis=1) > 0)[0]
    n, T, F = x.shape
    epochs = []
    if arm == "multinomial":
        y = d["A_onehot"]
        freq = (y * w[..., None]).sum(axis=(0, 1)) / max(w.sum(), 1.0)
        utils.set_seed(utils.sub_seed("treatment", arm))
        model = utils.build_sequence_model(T, F, J, "softmax", out_bias=np.log(freq + 1e-6))
        epochs.append(_fit(model, x, y, w, rows, utils.sub_seed("split", "treatment"),
                           HP["seq_max_epochs"], HP["seq_patience"]))
        probs = test_lstm.predict_sequence(model, x)
    else:
        probs = np.zeros((n, T, J), dtype="float32")
        for j in range(J):
            y = d["A_onehot"][..., j:j + 1]
            p = float((y[..., 0] * w).sum() / max(w.sum(), 1.0))
            utils.set_seed(utils.sub_seed("treatment", arm, j))
            model = utils.build_sequence_model(T, F, 1, "sigmoid", out_bias=[utils.logit(p)])
            epochs.append(_fit(model, x, y, w, rows, utils.sub_seed("split", "treatment"),
                               HP["seq_max_epochs"], HP["seq_patience"]))
            probs[..., j] = test_lstm.predict_sequence(model, x)[..., 0]
    return probs, np.asarray(epochs, dtype="int32")


def fit_censoring():
    """Returns (P(C_t = 0) (n, T), epochs); t = 0 is not modelled (no censoring at baseline)."""
    d = STATE["data"]
    x = _censoring_inputs()
    w = d["at_risk"].copy()
    w[:, 0] = 0.0
    y = (d["C"] == 0).astype("float32")[..., None]
    rows = np.where(w.sum(axis=1) > 0)[0]
    n, T, F = x.shape
    p = float((y[..., 0] * w).sum() / max(w.sum(), 1.0))
    utils.set_seed(utils.sub_seed("censoring"))
    model = utils.build_sequence_model(T, F, 1, "sigmoid", out_bias=[utils.logit(p)])
    ep = _fit(model, x, y, w, rows, utils.sub_seed("split", "censoring"), HP["seq_max_epochs"], HP["seq_patience"])
    return test_lstm.predict_sequence(model, x)[..., 0], np.asarray([ep], dtype="int32")


def _calibrate_intercept(model, x, z, iters=25):
    """Shift the output-layer bias (logit scale) so that the mean prediction among the fitted subjects
    equals the mean pseudo-outcome, as the intercept of a logistic regression guarantees. Without it,
    early-stopped networks warm-started from the previous (shorter-horizon) step can drift in level
    across the backward recursion."""
    if len(z) == 0:
        return
    p = np.clip(test_lstm.predict_sequence(model, x)[:, 0].astype("float64"), 1e-7, 1 - 1e-7)
    eta = np.log(p / (1 - p))
    target = float(np.mean(z))
    delta = 0.0
    for _ in range(iters):  # Newton steps for mean(expit(eta + delta)) = mean(z)
        q = 1 / (1 + np.exp(-(eta + delta)))
        g = q.mean() - target
        h = (q * (1 - q)).mean()
        if abs(g) < 1e-10 or h <= 0:
            break
        delta -= g / h
    w = model.get_weights()
    w[-1] = w[-1] + np.float32(delta)
    model.set_weights(w)


def outcome_model():
    """One compiled outcome model per process; its weights are swapped between fits (no retracing)."""
    if STATE["outcome_model"] is None:
        d = STATE["data"]
        F = d["base"].shape[-1] + d["J"]
        utils.set_seed(utils.sub_seed("outcome_init"))
        STATE["outcome_model"] = utils.build_outcome_model(int(HP["outcome_window"]), F)
        STATE["outcome_init_weights"] = STATE["outcome_model"].get_weights()
    return STATE["outcome_model"]


def fit_outcome_step(key, fallback_key, s, rows, z):
    """Fit the ICE regression of z on history ending at s among subjects `rows` (0-based).
    Warm-starts from the last fit stored under `key` (or `fallback_key`); identical fits
    (same s, rows and z, e.g. the first step Z = Y_t shared by all rules) are reused.
    Returns the number of epochs run (0 if reused)."""
    s = int(s)
    rows = np.asarray(rows, dtype="int64")
    z = np.asarray(z, dtype="float32")
    cache_key = utils.hash_arrays(np.asarray([s]), rows, z)
    if cache_key in STATE["fit_cache"]:
        STATE["warm"][key] = STATE["fit_cache"][cache_key]
        STATE["n_cached"] += 1
        return 0
    model = outcome_model()
    warm = STATE["warm"].get(key, STATE["warm"].get(fallback_key))
    if warm is not None:
        model.set_weights(warm)
        max_epochs, patience = HP["outcome_warm_max_epochs"], HP["outcome_warm_patience"]
    else:
        w0 = [w.copy() for w in STATE["outcome_init_weights"]]
        w0[-1] = np.full_like(w0[-1], utils.logit(float(np.mean(z)) if len(z) else 0.5))
        model.set_weights(w0)
        max_epochs, patience = HP["outcome_cold_max_epochs"], HP["outcome_cold_patience"]
    utils.reset_optimizer(model)
    x = test_lstm.outcome_window(s, rows)
    y = z[:, None]
    wt = np.ones(len(rows), dtype="float32")
    utils.set_seed(utils.sub_seed("outcome_fit", key, s))
    idx = np.arange(len(rows))
    _calibrate_intercept(model, x, z)  # warm start: move to the level of the new pseudo-outcome
    ep = _fit(model, x, y, wt, idx, utils.sub_seed("split", key, s), max_epochs, patience)
    _calibrate_intercept(model, x, z)  # mean prediction = mean pseudo-outcome among the fitted subjects
    weights = model.get_weights()
    STATE["warm"][key] = weights
    STATE["fit_cache"][cache_key] = weights
    STATE["fit_cache_order"].append(cache_key)
    while len(STATE["fit_cache_order"]) > 16:
        STATE["fit_cache"].pop(STATE["fit_cache_order"].pop(0), None)
    STATE["n_fits"] += 1
    return ep
