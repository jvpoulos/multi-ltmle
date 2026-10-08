"""
Prediction with the LSTM learners (rebuilt 2026-10).

  predict_sequence(model, x)                    many-to-many probabilities at every (subject, t)
  outcome_window(s, rows, a_override)           ICE input: last HP["outcome_window"] months of each
                                                subject's own history ending at s, left-padded with zeros
                                                (masked) when s < window - 1; treatment at s optionally
                                                replaced (counterfactual prediction under a rule)
  predict_outcome_step(key, s, rows, a_override) Q_s predictions from the fit stored under `key`
"""

import numpy as np

import utils
from utils import HP, STATE


def predict_sequence(model, x):
    return utils.predict_in_chunks(model, np.asarray(x, dtype="float32"))


def outcome_window(s, rows, a_override=None):
    d = STATE["data"]
    s = int(s)
    rows = np.asarray(rows, dtype="int64")
    W = int(HP["outcome_window"])
    k0 = max(0, s - W + 1)
    L = s - k0 + 1
    feats = np.concatenate([d["base"][rows, k0:s + 1, :], d["A_onehot"][rows, k0:s + 1, :]], axis=-1)
    if a_override is not None:
        F = d["base"].shape[-1]
        feats[:, -1, F:] = utils.onehot(np.asarray(a_override).reshape(-1), d["J"])
    x = np.zeros((len(rows), W, feats.shape[-1]), dtype="float32")
    x[:, W - L:, :] = feats
    return x


def predict_outcome_step(key, s, rows, a_override=None):
    import train_lstm
    model = train_lstm.outcome_model()
    model.set_weights(STATE["warm"][key])
    x = outcome_window(s, rows, a_override)
    return utils.predict_in_chunks(model, x)[:, 0]
