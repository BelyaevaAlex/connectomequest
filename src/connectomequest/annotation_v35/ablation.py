"""Third round of review: ablation of the recalibration and other earlier-release priors.

The frozen engine is untouched. ``AblatedRecalibration`` repeats the within-session
recalibration of :class:`engine.RecalibratedGreedy` with three of its choices exposed:
the shrinkage strength ``kappa`` (4 in the frozen rule), the edges of the groups of
initial membership probability (0.05, 0.25, 0.75, 0.95), and how the unresolved rate
of a group is shrunk: toward the average of the cell's prior unresolved mass and the
session rate ("pooled", frozen), toward the prior mass only ("prior") or toward the
session rate only ("session"). With the frozen choices it reproduces the frozen rule
exactly.

``earlier_release_parts`` recovers, from the stored earlier-release and topology priors,
which cells carry an earlier label and what that label says about a partner set; the
earlier-release prior of the paper is ``(1 - 0.9) * topology + 0.9 * label`` on those
cells and the topology prior elsewhere. ``earlier_release_prior`` rebuilds it with any
weight ``alpha`` or with class rates in place of the one-hot label.
"""

from __future__ import annotations

import numpy as np

from . import engine as e

FROZEN_EDGES = (0.05, 0.25, 0.75, 0.95)
FROZEN_ALPHA = 0.9
CLASSES = ("partner", "outside", "other")


class AblatedRecalibration(e.AdaptiveGreedy):
    """Adaptive Greedy over beliefs recalibrated within the session, with exposed choices."""

    def __init__(self, instance, probabilities, *, kappa=4.0, edges=FROZEN_EDGES, unresolved="pooled"):
        super().__init__(instance, probabilities)
        if instance.outcomes != 3:
            raise ValueError("recalibration expects member/nonmember/unresolved outcomes")
        if unresolved not in ("pooled", "prior", "session"):
            raise ValueError("unknown treatment of the unresolved rate")
        resolved = np.maximum(self.p[:, e.MEMBER] + self.p[:, e.NONMEMBER], 1e-12)
        self._prior_member = self.p[:, e.MEMBER] / resolved
        self._prior_unresolved = self.p[:, e.UNRESOLVED].copy()
        self._edges = np.asarray(edges, dtype=float)
        self._group = np.searchsorted(self._edges, self._prior_member, side="right")
        self._counts = np.zeros((len(self._edges) + 1, 3))
        self._strength = float(kappa)
        self._all = np.zeros(3)
        self._mode = unresolved

    def observe(self, cell, outcome):
        self._counts[self._group[cell], outcome] += 1
        self._all[outcome] += 1
        counts = self._counts[self._group]
        resolved = counts[:, e.MEMBER] + counts[:, e.NONMEMBER]
        total = resolved + counts[:, e.UNRESOLVED]
        session_unresolved = (self._all[e.UNRESOLVED] + 0.5) / (self._all.sum() + 1.0)
        keep = self._strength / (self._strength + resolved)
        member = keep * self._prior_member + (1 - keep) * counts[:, e.MEMBER] / np.maximum(resolved, 1)
        keep_u = self._strength / (self._strength + total)
        if self._mode == "pooled":
            prior_u = 0.5 * self._prior_unresolved + 0.5 * session_unresolved
        elif self._mode == "prior":
            prior_u = self._prior_unresolved
        else:
            prior_u = np.full_like(self._prior_unresolved, session_unresolved)
        unresolved = keep_u * prior_u + (1 - keep_u) * counts[:, e.UNRESOLVED] / np.maximum(total, 1)
        self.p[:, e.MEMBER] = (1 - unresolved) * member
        self.p[:, e.NONMEMBER] = (1 - unresolved) * (1 - member)
        self.p[:, e.UNRESOLVED] = unresolved


RECAL_VARIANTS = {
    "frozen": dict(),
    "kappa1": dict(kappa=1.0),
    "kappa16": dict(kappa=16.0),
    "kappa64": dict(kappa=64.0),
    "groups1": dict(edges=()),
    "groups3": dict(edges=(0.25, 0.75)),
    "groups8": dict(edges=(0.02, 0.05, 0.25, 0.5, 0.75, 0.95, 0.98)),
    "uprior": dict(unresolved="prior"),
    "usession": dict(unresolved="session"),
}


def earlier_release_parts(plus_earlier, unknown_earlier, plus_topology, unknown_topology, alpha=FROZEN_ALPHA, tolerance=1e-4):
    """Cells with an earlier label and the class of that label for one partner set.

    Returns ``(labelled, cls)``: ``labelled`` marks cells with an earlier label; ``cls`` is
    0 when the label is a partner type, 1 when it lies outside the vocabulary, 2 for
    another hemibrain type (and for cells without a label). Raises if the stored prior
    is not the documented mixture.
    """
    pe, ue = np.asarray(plus_earlier, dtype=np.float64), np.asarray(unknown_earlier, dtype=np.float64)
    pt, ut = np.asarray(plus_topology, dtype=np.float64), np.asarray(unknown_topology, dtype=np.float64)
    same = (np.abs(pe - pt) <= 1e-7) & (np.abs(ue - ut) <= 1e-7)
    label_plus = (pe - (1 - alpha) * pt) / alpha
    label_unknown = (ue - (1 - alpha) * ut) / alpha
    labelled = ~same
    for value in (label_plus[labelled], label_unknown[labelled]):
        if value.size and not np.all((np.abs(value) <= tolerance) | (np.abs(value - 1) <= tolerance)):
            raise ValueError("stored earlier-release prior is not the documented mixture")
    partner = labelled & (label_plus > 0.5)
    outside = labelled & (label_unknown > 0.5)
    if np.any(partner & outside):
        raise ValueError("a label cannot be both a partner type and outside the vocabulary")
    cls = np.where(partner, 0, np.where(outside, 1, 2))
    return labelled, cls


def earlier_release_prior(topology, labelled, cls, *, alpha=None, rates=None):
    """Rebuild the earlier-release prior: mixture weight ``alpha`` or class ``rates`` (3 x 3)."""
    q = np.asarray(topology, dtype=float).copy()
    onehot = np.zeros_like(q)
    onehot[cls == 0, e.MEMBER] = 1.0
    onehot[cls == 1, e.UNRESOLVED] = 1.0
    onehot[cls == 2, e.NONMEMBER] = 1.0
    if (alpha is None) == (rates is None):
        raise ValueError("give either a mixture weight or class rates")
    if alpha is not None:
        q[labelled] = (1 - alpha) * q[labelled] + alpha * onehot[labelled]
    else:
        rates = np.asarray(rates, dtype=float)
        q[labelled] = rates[cls[labelled]]
    return q


def class_rates(classes, outcomes):
    """Observed shares of (member, nonmember, unresolved) per class of earlier label."""
    classes, outcomes = np.asarray(classes), np.asarray(outcomes)
    rates = np.zeros((len(CLASSES), 3))
    for k in range(len(CLASSES)):
        y = outcomes[classes == k]
        if not len(y):
            raise ValueError("a class without cells")
        rates[k] = np.bincount(y, minlength=3)[:3] / len(y)
    return rates
