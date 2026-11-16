"""Two generic policies for threshold questions: decision region determination and information gain.

Both treat the prior as a product distribution over the outcomes of the unchecked cells and ask how a check would
change the probabilities of the three possible decisions about the question: answer *yes*, answer *no*, or report
that neither can be certified. The decision probabilities are estimated from ``samples`` realizations of the
unchecked outcomes. Because the prior is a product, a check of one cell does not change the distribution of the
other cells, so the same uniform draws serve after every check (common random numbers): only the probabilities
they are compared with may change, which is how the recalibrated variants follow the recalibrated beliefs.

* ``EdgeCutting`` is the greedy policy of decision region determination for disjoint regions, equivalence class
  edge cutting (Golovin, Krause and Ray 2010; Javdani et al. 2014): hypotheses are realizations, two realizations
  with different decisions are joined by an edge weighted by the product of their probabilities, and the policy
  checks the cell that cuts the largest expected edge weight. With decision probabilities ``pi`` the weight that
  remains is proportional to ``G = sum_{i<j} pi_i pi_j``; after outcome ``o`` of probability ``p_o`` it is
  ``p_o^2 G_o``, so the policy minimizes ``sum_o p_o^3 G_o``.
* ``InformationGain`` checks the cell whose outcome has the largest mutual information with the decision, the
  criterion that active feature acquisition estimates with learned models (Ma et al. 2019; Covert et al. 2023),
  computed here from the prior.

Both methods come with their own stopping rule: stop when every hypothesis still considered leads to one
decision. Here a certificate needs the exact bounds, so the policies keep checking; ``unanimous`` records the first
moment at which all sampled realizations agreed on a decision, and that decision, so that the stopping rule can be
judged against the reference (``(checks made, decision)`` with decisions 0 uncertifiable, 1 yes, 2 no).

Probabilities only order checks; every cost is checked by the exact verifier of ``engine``.
"""

from __future__ import annotations

import numpy as np

from . import engine as e

SAMPLES = 256


class _Sampled(e._Adaptive):
    """Decision probabilities after every outcome of every unchecked cell, from sampled realizations."""

    def __init__(self, instance, probabilities):
        super().__init__(instance, probabilities)
        if instance.outcomes != 3:
            raise ValueError("member/nonmember/unresolved outcomes expected")
        self._u = None
        self.unanimous = None

    def seed(self, rng, samples=SAMPLES):
        """Draw the uniform numbers that define the sampled realizations; returns the policy."""
        self._u = rng.random((int(samples), self.instance.size))
        return self

    def score(self, p, now, after):
        raise NotImplementedError

    def decisions(self, a, b, available):
        """(indices, p, now, after): ``now`` = (yes, no) probabilities, ``after[o]`` = (yes, no) arrays per cell."""
        idx = np.flatnonzero(available)
        p = self.p[idx]
        u = self._u[:, idx]
        member = u < p[:, e.MEMBER]
        nonmember = ~member & (u < p[:, e.MEMBER] + p[:, e.NONMEMBER])
        w = self._w[idx]
        member_weight, nonmember_weight = member * w, nonmember * w
        total_member, total_nonmember = member_weight.sum(1), nonmember_weight.sum(1)
        now = (float((total_member >= a).mean()), float((total_nonmember >= b).mean()))
        without_member = total_member[:, None] - member_weight
        without_nonmember = total_nonmember[:, None] - nonmember_weight
        yes_if_member = (without_member + w >= a).mean(0)
        yes_otherwise = (without_member >= a).mean(0)
        no_if_nonmember = (without_nonmember + w >= b).mean(0)
        no_otherwise = (without_nonmember >= b).mean(0)
        after = ((yes_if_member, no_otherwise), (yes_otherwise, no_if_nonmember), (yes_otherwise, no_otherwise))
        return idx, p, now, after

    def _pick(self, idx, score):
        """Largest score, ties by ascending external ID, as in ``engine``."""
        return int(idx[np.lexsort((self.instance.ranks[idx], -score))[0]])

    def choose(self, a, b, available):
        idx, p, now, after = self.decisions(a, b, available)
        if self.unanimous is None and (now[0] == 1.0 or now[1] == 1.0 or now[0] + now[1] == 0.0):
            self.unanimous = (self.instance.size - len(idx), 1 if now[0] == 1.0 else 2 if now[1] == 1.0 else 0)
        score = self.score(p, now, after)
        if not np.any(score > 1e-12):
            return self._fallback(available)
        return self._pick(idx, score)


def _edges(yes, no):
    """Sum over pairs of distinct decisions of the product of their probabilities."""
    neither = 1.0 - yes - no
    return yes * no + yes * neither + no * neither


def _entropy(yes, no):
    out = 0.0
    for q in (yes, no, 1.0 - yes - no):
        q = np.clip(q, 0.0, 1.0)
        out = out - np.where(q > 0, q * np.log(np.where(q > 0, q, 1.0)), 0.0)
    return out


class EdgeCutting(_Sampled):
    """Equivalence class edge cutting over the decisions yes / no / uncertifiable."""

    def score(self, p, now, after):
        remaining = sum(p[:, o] ** 3 * _edges(*after[o]) for o in range(3))
        return _edges(*now) - remaining


class InformationGain(_Sampled):
    """Largest mutual information between the outcome of a check and the decision."""

    def score(self, p, now, after):
        expected = sum(p[:, o] * _entropy(*after[o]) for o in range(3))
        return _entropy(*now) - expected


class RecalibratedEdgeCutting(e._Recalibration, EdgeCutting):
    """Edge cutting over beliefs recalibrated within the question (the recalibration of ``engine``)."""


class RecalibratedInformationGain(e._Recalibration, InformationGain):
    """Information gain over beliefs recalibrated within the question (the recalibration of ``engine``)."""


POLICIES = dict(ec2=EdgeCutting, ig=InformationGain, **{"ec2-recal": RecalibratedEdgeCutting, "ig-recal": RecalibratedInformationGain})


def policy(name, instance, probabilities, rng, samples=SAMPLES):
    return POLICIES[name](instance, probabilities).seed(rng, samples)
