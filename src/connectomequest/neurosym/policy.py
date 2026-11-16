"""Pilot A: a linear index policy whose parameters are fitted to certificate cost.

``IndexPolicy`` pays the available cell with the largest ``theta . phi_j(state)``
(ties: ascending external ID, as in the frozen engine). The features reuse the
frozen engine's own arithmetic, so that the engine policies are exact special
cases of ``theta`` (see ``SPECIAL``):

* ``gain``        two-goal expected utility gain of Adaptive Greedy, divided by ``a*b``
                  (``= p_m min(w,a)/a + p_n min(w,b)/b``);
* ``pos``         capped expected progress toward the positive goal, ``p_m min(w,a)/a``;
* ``dir_pos``     ``pos`` if the positive goal is forecast to need fewer further
                  payments (Adaptive Direction's forecast), else 0;
* ``dir_neg``     ``p_n min(w,b)/b`` if the negative goal is forecast closer, else 0;
* ``static_pos``  frozen initial direction (``direction_order``): uncapped initial
                  ``p_m w / a0`` if the positive order was chosen at the start, else 0;
* ``static_neg``  the same for the negative order;
* ``cap``         ``(min(w,a)/a + min(w,b)/b) / 2``, a prior-free capped weight;
* ``logw``        ``log w / log W``;
* ``unresolved``  current probability of an unresolved response.

Beliefs are ``(1 - recal) * prior + recal * recalibrated`` where the recalibrated
beliefs are the frozen engine's in-session recalibration with ``strength``.
If no available cell has a positive index the policy pays the heaviest cell
(the engine's fallback); ``fallback=False`` pays by the index (ties by ID) instead,
which is what the static ``direction_order`` does.

Probabilities only order the payments; every cost is checked by the exact verifier.
"""

from __future__ import annotations

import numpy as np

from connectomequest.annotation_v35 import engine as e

FEATURES = ("gain", "pos", "dir_pos", "dir_neg", "static_pos", "static_neg", "cap", "logw", "unresolved")


def _theta(**kw):
    t = np.zeros(len(FEATURES))
    for k, v in kw.items():
        t[FEATURES.index(k)] = v
    return t


# name -> (theta, recal, strength, fallback); engine policies as special cases
SPECIAL = {
    "ag": (_theta(gain=1.0), 0.0, 4.0, True),
    "recal": (_theta(gain=1.0), 1.0, 4.0, True),
    "adir": (_theta(dir_pos=1.0, dir_neg=1.0), 0.0, 4.0, True),
    "adir-recal": (_theta(dir_pos=1.0, dir_neg=1.0), 1.0, 4.0, True),
    "direction": (_theta(static_pos=1.0, static_neg=1.0), 0.0, 4.0, False),
}


class IndexPolicy(e._Adaptive):
    """``argmax theta . phi`` over available cells; see the module docstring."""

    def __init__(self, instance, probabilities, theta, *, recal=0.0, strength=4.0, fallback=True):
        super().__init__(instance, probabilities)
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (len(FEATURES),) or not np.isfinite(theta).all():
            raise ValueError("theta must be a finite vector over FEATURES")
        if not 0.0 <= float(recal) <= 1.0 or not float(strength) > 0:
            raise ValueError("recal in [0, 1] and positive strength required")
        self.theta = theta
        self.active = [(FEATURES[k], float(theta[k])) for k in range(len(FEATURES)) if theta[k] != 0.0]
        self.recal = float(recal)
        self.fallback_enabled = bool(fallback)
        self.prior = self.p.copy()
        self.tracker = e.RecalibratedGreedy(instance, probabilities, strength=float(strength)) if self.recal > 0 else None
        names = {n for n, _ in self.active}
        if names & {"static_pos", "static_neg"}:
            positive, negative = e.expected_progress(instance, self.prior)
            (_, fp), (_, fn) = e.direction_forecasts(instance, self.prior)
            zero = np.zeros(instance.size)
            self._static = dict(static_pos=positive / instance.a0 if fp <= fn else zero, static_neg=zero if fp <= fn else negative / instance.b0)
        if "logw" in names:
            self._logw = np.log(self._w) / np.log(max(float(self._w.sum()), 2.0))

    def observe(self, cell, outcome):
        if self.tracker is None:
            return
        self.tracker.observe(cell, outcome)
        self.p = self.tracker.p.copy() if self.recal == 1.0 else (1.0 - self.recal) * self.prior + self.recal * self.tracker.p

    def features(self, a, b, available):
        """Dict of the active features (all cells; only ``available`` ones are used)."""
        names = {n for n, _ in self.active}
        a, b = float(a), float(b)
        out = {}
        if "gain" in names:
            out["gain"] = self._gain(a, b) / (a * b)
        if names & {"pos", "dir_pos", "dir_neg"}:
            positive, negative = e.AdaptiveDirection._directional(self, a, b)
            if "pos" in names:
                out["pos"] = positive / a
            if names & {"dir_pos", "dir_neg"}:
                need_p = e.AdaptiveDirection._needed(positive[available], a)
                need_n = e.AdaptiveDirection._needed(negative[available], b)
                zero = np.zeros(len(positive))
                toward_positive = need_p <= need_n
                out["dir_pos"] = positive / a if toward_positive else zero
                out["dir_neg"] = zero if toward_positive else negative / b
        for k in ("static_pos", "static_neg"):
            if k in names:
                out[k] = self._static[k]
        if "cap" in names:
            out["cap"] = (np.minimum(self._w, a) / a + np.minimum(self._w, b) / b) / 2.0
        if "logw" in names:
            out["logw"] = self._logw
        if "unresolved" in names:
            out["unresolved"] = self.p[:, e.UNRESOLVED]
        return out

    def score(self, a, b, available):
        phi = self.features(a, b, available)
        score = np.zeros(self.instance.size)
        for name, coef in self.active:
            score = score + coef * phi[name]
        return score

    def choose(self, a, b, available):
        score = self.score(a, b, available)
        if self.fallback_enabled and not np.any(available & (score > 0)):
            return self._fallback(available)
        return self._best(score, available)


def make(instance, probabilities, params):
    """Policy from a parameter dict ``{theta, recal, strength, fallback}``."""
    return IndexPolicy(instance, probabilities, params["theta"], recal=params.get("recal", 0.0),
                       strength=params.get("strength", 4.0), fallback=params.get("fallback", True))


def special(name):
    theta, recal, strength, fallback = SPECIAL[name]
    return dict(theta=theta.copy(), recal=recal, strength=strength, fallback=fallback)


# ---- black-box parametrisation for CEM -------------------------------------------------

DIM = len(FEATURES) + 2  # theta, recal (clipped to [0, 1]), log2 strength (clipped to [-2, 8])


def decode(x):
    x = np.asarray(x, dtype=float)
    theta = x[: len(FEATURES)]
    norm = np.abs(theta).sum()
    theta = theta / norm if norm > 0 else _theta(gain=1.0)
    return dict(theta=theta, recal=float(np.clip(x[-2], 0.0, 1.0)), strength=float(2.0 ** np.clip(x[-1], -2.0, 8.0)), fallback=True)


def encode(params):
    theta = np.asarray(params["theta"], dtype=float)
    return np.concatenate((theta / np.abs(theta).sum(), [params["recal"], np.log2(params["strength"])]))


def run_session(instance, outcomes, probabilities, params):
    return e.run_adaptive(instance, outcomes, make(instance, probabilities, params))
