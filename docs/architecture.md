# Architecture

ConnectomeQuest separates four responsibilities:

1. **Environment:** owns the hidden graph or simulator state and exposes only declared observations.
2. **Learned component:** ranks graph candidates or maps RGB tiles to semantic claims.
3. **Symbolic controller:** checks action preconditions, tracks the acquisition budget, and selects the next action.
4. **Certificate verifier:** validates a submitted witness against trusted issuance records without access to the hidden graph.

The graph and embodied evaluations implement the same typed loop:

```text
observation -> learned proposal -> symbolic admissibility -> action
            <- receipt + next observation <- environment
certificate -> independent verifier -> accept / reject
```

## Graph implementation

`ConnectomeEnv` discloses bounded neighbor pages and charges acquisition actions.
Policies rank only candidates present in the common observation. The controller
executes the ordering and reserves one transition for submission. An answer is
successful only when the submitted two-hop witness is accepted.

## Embodied implementation

The agent receives a 7x7 egocentric RGB image, pose relative to its initial
position, cardinal direction, inventory type, mission text, and last-action
outcome. A frozen tile classifier proposes semantic-map claims. Each claim keeps
the identifier of the image receipt from which it was inferred. The action
model lists the claims required by each planned action.

Complete validation scans the retained plan after an evidence update.
Dependency-indexed validation records inverse maps from receipts and semantic
keys to affected plan positions. Both use the same planner and authorization
rule. Unchecked reuse is only a safety ablation.

## Trust boundary

Receipts prove that an observation was issued in a particular episode and
context. They do not prove that a learned interpretation is physically true.
The formal guarantees are therefore relative to the trusted issuer and the
symbolic action model.
