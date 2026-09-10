# Baselines and access regimes

Comparisons are grouped by what each method may observe.

## Active observable policies

Weight, Random, Recurrent, Attribute, Expert selector, and Random-to-Attribute
receive the same public observation and execute through the same budgeted
controller. Their results are directly comparable.

## Privileged ceilings

Visible-witness and full-graph oracles are diagnostic ceilings. They may use
information unavailable to active policies and are never presented as
access-matched competitors.

## Static graph-query models

Query2Box, BetaE, ConE, GNN-QE, UltraQuery, AStarNet, and NBFNet operate on a
static/full-graph access regime. Their archived diagnostics address a different
task and are excluded from the primary publication bundle.

## Embodied controls

Full replanning, complete validation, and dependency-indexed validation share
observations, action model, planner, and update stream. Unchecked reuse performs
no validation after an update and is included only to expose the safety failure
that validation prevents.
