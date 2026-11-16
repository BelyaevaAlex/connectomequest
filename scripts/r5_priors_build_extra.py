#!/usr/bin/env python3
"""Supplementary seeded priors at smaller seed fractions (1% and 3%), written to outputs/round5/priors/extra.

Identical to ``scripts/r5_priors_build.py`` (which it runs) except for the admissible fractions and the
default output directory; the files keep the layout and naming (``seeded01``, ``seeded03conn`` ...). They
locate how much confirmed curation the in-connectome predictor needs before it reaches the AUROC at which
predictions pay, and are kept apart so that consumers of the main priors do not pick them up.
"""

from __future__ import annotations

import sys

from connectomequest.round5 import priors_common as pc

EXTRA_FRACTIONS = (0.01, 0.03)


def main():
    pc.FRACTIONS = EXTRA_FRACTIONS
    if "--output" not in sys.argv:
        sys.argv += ["--output", str(pc.OUT / "extra")]
    import r5_priors_build

    r5_priors_build.main()


if __name__ == "__main__":
    main()
