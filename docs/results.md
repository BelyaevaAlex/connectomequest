# Frozen results

These numbers are generated from the same records used by the manuscript. The
LaTeX tables in `paper/generated/` are the canonical rendered source.

## Active graph acquisition

Certificate success / unconditional charged actions:

| Graph | Weight | Random | Recurrent | Attribute | Expert selector |
|---|---:|---:|---:|---:|---:|
| H01 | .4090 / 42.63 | .5100 / 39.82 | .4840 / 40.64 | .4117 / 41.27 | .3920 / 43.14 |
| MANC | .0990 / 60.18 | .1050 / 60.11 | .1030 / 60.01 | .1107 / 59.96 | .1120 / 60.03 |
| HemiBrain | .1270 / 59.52 | .1347 / 59.23 | .1437 / 58.86 | .1343 / 58.91 | .1483 / 58.83 |

Random is strongest on H01, while the expert selector and recurrent policy are
strongest on the two fly-graph rows respectively. The result supports policy
complementarity and an information-acquisition bottleneck; it does not support
a universal learned-policy superiority claim.

## Evidence-indexed plan validation

On 221 eligible long-plan layouts and 1,326 paired timing cells,
dependency-indexed and complete validation agree on 100% of authorization
decisions. Dependency indexing reduces the measured validation/replanning CPU
endpoint by 96.7% with a layout-bootstrap interval of [96.4%, 96.9%]. Under
final-support loss, full replanning, complete validation, and dependency-indexed
validation make 0/513 unsupported authorizations; unchecked reuse makes 511/513.

## Certificate audit

The verifier accepts all 22,662 unmodified nonempty certificates from 60,000
graph episodes. It rejects all 13,800 generated provenance/binding and
structural mutations. These are coverage counts for the stated generators, not
an estimated real-world attack success probability.
