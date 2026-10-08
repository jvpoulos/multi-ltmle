"""
Shared utilities for the LSTM learners used by the LTMLE core (rebuilt 2026-10).

The R bridge (src/lstm.R) sends one replicate's per-subject arrays once per process
(set_data); the learners in train_lstm.py / test_lstm.py then build model inputs from
them. Every array row is one subject and every time index is that subject's own month,
so no input sequence ever spans two subjects, and predictions are returned keyed by
(row = subject, time). All LSTMs are unidirectional and the optional self-attention is
causally masked, so a prediction at time t depends on history up to t only.

Arrays sent from R (T = K + 1 time points t = 0..K):
  base   (n, T, F)  baseline and covariate features at each t:
                    V1 one-hot (3), V2 one-hot (2), V3 (robust-scaled), log1p(L1_t), L2_t, L3_t, t / t_end
  A      (n, T)     observed treatment 1..J (0 if unobserved)
  C      (n, T)     censoring indicator (1 = censored, 0 = uncensored, 0 if unobserved)
  at_risk(n, T)     1 if uncensored through t-1 and no event through t-1
"""

import os
import random
import hashlib
import warnings

import numpy as np

# the last-time-step slice of the (left-padded) outcome window drops the padding mask on purpose
warnings.filterwarnings("ignore", message=".*does not support masking and will therefore destroy the mask.*")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# hyperparameters actually used (reported in the SI); overridable from R via configure(hp=...)
HP = {
    "units": 32,                 # LSTM units per layer
    "dense_units": 16,           # pre-output dense layer (ReLU)
    "input_skip": True,          # residual (skip) connection from the current month's inputs to the output head
    "dropout": 0.1,
    "weight_decay": 1e-4,        # AdamW
    "clipnorm": 1.0,
    "batch_size": 128,
    "val_fraction": 0.1,         # subject-level validation split for early stopping
    # treatment / censoring many-to-many models (fit once per replicate)
    "seq_n_lstm_layers": 2,      # stacked LSTM layers (residual connection between them)
    "seq_attention": True,       # single-head, causally masked self-attention after the LSTM stack
    "seq_learning_rate": 1e-3,
    "seq_max_epochs": 30,
    "seq_patience": 3,
    # outcome ICE regressions (one fit per step of each recursion)
    "outcome_window": 12,        # months of history (ending at s) used by the outcome regressions
    "outcome_n_lstm_layers": 1,
    "outcome_attention": False,
    "outcome_learning_rate": 3e-3,
    "outcome_cold_max_epochs": 20,
    "outcome_cold_patience": 3,
    "outcome_warm_max_epochs": 6,
    "outcome_warm_patience": 2,
    "predict_batch": 8192,
}

HP_DEFAULT = dict(HP)

# per-process state: data of the current replicate, compiled outcome model, warm-start weights
STATE = {"tf": None, "keras": None, "data_token": None, "data": None, "seed": 0,
         "outcome_model": None, "outcome_init_weights": None, "warm": {}, "fit_cache": {},
         "fit_cache_order": [], "n_fits": 0, "n_cached": 0}


def configure(threads=1, seed=0, hp=None):
    """Import TensorFlow/Keras in this process, set threads and seeds, and set hyperparameters
    (the defaults, overridden by the entries of `hp`)."""
    HP.clear()
    HP.update(HP_DEFAULT)
    if hp:
        for k, v in dict(hp).items():
            if k not in HP:
                raise ValueError("unknown LSTM hyperparameter: %s" % k)
            HP[k] = bool(v) if isinstance(HP[k], bool) else type(HP[k])(v)
    if STATE["tf"] is None:
        import tensorflow as tf
        try:
            tf.config.threading.set_intra_op_parallelism_threads(int(threads))
            tf.config.threading.set_inter_op_parallelism_threads(int(threads))
        except RuntimeError:
            pass  # threads can only be set before TF initialises; keep the existing setting
        import keras
        STATE["tf"], STATE["keras"] = tf, keras
    set_seed(seed)
    return True


def set_seed(seed):
    seed = int(seed) % (2 ** 31 - 1)
    STATE["seed"] = seed
    random.seed(seed)
    np.random.seed(seed)
    if STATE["keras"] is not None:
        STATE["keras"].utils.set_random_seed(seed)


def sub_seed(*keys):
    """Deterministic seed derived from the process seed and a tuple of keys."""
    h = hashlib.md5(repr((STATE["seed"],) + tuple(keys)).encode()).hexdigest()
    return int(h[:8], 16)


def set_data(token, base, A, C, at_risk, J=6):
    """Store one replicate's arrays (once per process and replicate)."""
    if STATE["data_token"] == token:
        return False
    base = np.asarray(base, dtype="float32")
    A = np.asarray(A).astype("int32")
    STATE["data"] = {
        "base": base,
        "A": A,
        "C": np.asarray(C).astype("int32"),
        "at_risk": np.asarray(at_risk).astype("float32"),
        "A_onehot": onehot(A, J),
        "J": int(J),
    }
    STATE["data_token"] = token
    STATE["outcome_model"] = None
    STATE["outcome_init_weights"] = None
    STATE["warm"] = {}
    STATE["fit_cache"] = {}
    STATE["fit_cache_order"] = []
    return True


def onehot(A, J=6):
    """One-hot encode treatments 1..J; 0 (unobserved) maps to all zeros."""
    A = np.asarray(A).astype("int32")
    out = np.zeros(A.shape + (J,), dtype="float32")
    for j in range(1, J + 1):
        out[..., j - 1] = (A == j)
    return out


def lagged(x):
    """Shift a (n, T, ...) array one step forward in time (zeros at t = 0)."""
    out = np.zeros_like(x)
    out[:, 1:] = x[:, :-1]
    return out


def split_rows(rows, frac, seed):
    """Subject-level train/validation split of row indices."""
    rng = np.random.default_rng(seed)
    rows = np.asarray(rows)
    if frac <= 0 or len(rows) < 50:
        return rows, rows[:0]
    perm = rng.permutation(len(rows))
    n_val = max(1, int(round(frac * len(rows))))
    return np.sort(rows[perm[n_val:]]), np.sort(rows[perm[:n_val]])


def _optimizer(learning_rate):
    keras = STATE["keras"]
    return keras.optimizers.AdamW(learning_rate=float(learning_rate), weight_decay=HP["weight_decay"],
                                  clipnorm=HP["clipnorm"])


def _lstm_stack(x, n_layers, attention):
    """Stacked unidirectional LSTMs with layer normalisation, a residual connection,
    and optional single-head causally masked self-attention. Returns the sequence output.
    Keras propagates any padding mask from a Masking layer through these layers."""
    keras = STATE["keras"]
    layers = keras.layers
    h = layers.LSTM(HP["units"], return_sequences=True)(x)
    h = layers.LayerNormalization()(h)
    for _ in range(int(n_layers) - 1):
        h2 = layers.LSTM(HP["units"], return_sequences=True)(h)
        h = layers.LayerNormalization()(layers.Add()([h, h2]))
    if attention:
        att = layers.MultiHeadAttention(num_heads=1, key_dim=HP["units"])(h, h, use_causal_mask=True)
        h = layers.LayerNormalization()(layers.Add()([h, att]))
    if HP["dropout"] > 0:
        h = layers.Dropout(HP["dropout"])(h)
    return h


def build_sequence_model(T, F, n_out, activation, out_bias=None):
    """Many-to-many model: an output at every time t from history up to t."""
    keras = STATE["keras"]
    layers = keras.layers
    inp = keras.Input(shape=(T, F))
    h = _lstm_stack(inp, HP["seq_n_lstm_layers"], HP["seq_attention"])
    h = layers.Dense(HP["dense_units"], activation="relu")(h)
    if HP["input_skip"]:
        h = layers.Concatenate()([h, inp])  # shallow features of month t alongside the deep representation
    init = keras.initializers.Constant(out_bias) if out_bias is not None else "zeros"
    # zero output kernel: training starts from the marginal frequencies (calibrated in the large)
    out = layers.Dense(n_out, activation=activation, kernel_initializer="zeros", bias_initializer=init)(h)
    model = keras.Model(inp, out)
    loss = "categorical_crossentropy" if activation == "softmax" else "binary_crossentropy"
    model.compile(optimizer=_optimizer(HP["seq_learning_rate"]), loss=loss)
    return model


def build_outcome_model(W, F):
    """Sequence-to-scalar model for one ICE regression: history window of W months ending at s
    (left-padded with zeros, masked) -> Q_s in (0, 1). Uses the last hidden state."""
    keras = STATE["keras"]
    layers = keras.layers
    inp = keras.Input(shape=(W, F))
    h = _lstm_stack(layers.Masking(mask_value=0.0)(inp), HP["outcome_n_lstm_layers"], HP["outcome_attention"])
    last = layers.Lambda(lambda z: z[:, -1, :])(h)
    last = layers.Dense(HP["dense_units"], activation="relu")(last)
    if HP["input_skip"]:
        # shallow features of month s (covariates and treatment at s) alongside the deep representation
        last = layers.Concatenate()([last, layers.Lambda(lambda z: z[:, -1, :])(inp)])
    out = layers.Dense(1, activation="sigmoid", kernel_initializer="zeros")(last)
    model = keras.Model(inp, out)
    model.compile(optimizer=_optimizer(HP["outcome_learning_rate"]), loss=keras.losses.BinaryCrossentropy())
    return model


def reset_optimizer(model):
    """Restore the optimizer state (iteration count, moments; learning rate unchanged) captured right
    after the optimizer was built, so each fit starts fresh without recompiling the model."""
    opt = model.optimizer
    if getattr(model, "_opt_init_values", None) is None:
        opt.build(model.trainable_variables)
        model._opt_init_values = [np.array(v.numpy()) for v in opt.variables]
    for v, init in zip(opt.variables, model._opt_init_values):
        v.assign(init)


def logit(p):
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    return float(np.log(p / (1 - p)))


def predict_in_chunks(model, x):
    out = []
    b = int(HP["predict_batch"])
    for i in range(0, x.shape[0], b):
        out.append(np.asarray(model(x[i:i + b], training=False)))
    return np.concatenate(out, axis=0) if out else np.zeros((0,) + tuple(model.output_shape[1:]), dtype="float32")


def hash_arrays(*arrays):
    h = hashlib.md5()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()
