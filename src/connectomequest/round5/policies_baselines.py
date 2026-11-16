"""Generic active-learning baselines (round 5, workstream ``policies``).

* **Uncertainty sampling** (``us``): check the unchecked cell whose answer is most
  uncertain, i.e. with the largest Shannon entropy of ``(p+, p-, p_unres)``.
* **Weight-scaled uncertainty sampling** (``wus``): the largest ``w_v * entropy``.

With a static prior both are fixed orders. The ``-recal`` variants re-score the
unchecked cells after every answer with the in-question recalibration of the frozen
engine (the analogue of refitting the model in pool-based active learning). Ties
are broken by ascending neuron identifier, as for every frozen policy. The stopping
rule and the verifier are those of the frozen engine; probabilities only order checks.
"""

from __future__ import annotations

import numpy as np

from connectomequest.annotation_v35 import engine as e


def entropy(p):
    """Shannon entropy (nats) of every row; ``0 log 0 = 0``."""
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, -p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    return terms.sum(1)


def uncertainty_order(instance, probabilities):
    return e._descending(instance, entropy(e.check_probabilities(instance, probabilities)))


def weighted_uncertainty_order(instance, probabilities):
    p = e.check_probabilities(instance, probabilities)
    return e._descending(instance, instance.weights.astype(float) * entropy(p))


class _Uncertainty(e._Adaptive):
    scaled = False

    def choose(self, a, b, available):
        score = entropy(self.p)
        if self.scaled:
            score = score * self._w
        return self._best(score, available)


class RecalUncertainty(e._Recalibration, _Uncertainty):
    """Uncertainty sampling over beliefs recalibrated within the question."""


class RecalWeightedUncertainty(e._Recalibration, _Uncertainty):
    """Weight-scaled uncertainty sampling over recalibrated beliefs."""

    scaled = True


ORDERS = {"us": uncertainty_order, "wus": weighted_uncertainty_order}
ADAPTIVE = {"us-recal": RecalUncertainty, "wus-recal": RecalWeightedUncertainty}


def run(name, instance, outcomes, probabilities):
    if name in ORDERS:
        return ORDERS[name](instance, probabilities)
    return e.run_adaptive(instance, outcomes, ADAPTIVE[name](instance, probabilities))
