# Evaluation protocol

## Interaction process

An episode is defined by a hidden environment state, a public observation,
a legal-action relation, action costs, an acquisition budget, a controller,
and a certificate verifier. Policies never receive the hidden graph or answer
set.

The graph task asks for one node satisfying a typed two-hop query. The standard
setting uses visibility 32, budget 64, horizon 64, and no free probe. All
policies receive the same candidate lists and execute through the same
controller.

## Graph policies

- **Weight:** descending disclosed edge weight.
- **Random:** branch-diverse random ordering fixed by the episode seed.
- **Recurrent:** recurrent learned ordering without entity embeddings.
- **Attribute:** listwise ranker using observable candidate attributes.
- **Expert selector:** pre-action selector over fixed experts.
- **Random to Attribute:** random first-hop order and Attribute second-hop order.

Learned policies are trained on two connectomes and evaluated on the third.
Entity identifiers are not embedding-table indices. Numeric identifiers may
still affect deterministic tie ordering, which is documented rather than
claimed invariant.

## Certificate success

A graph episode succeeds when the agent submits a two-hop wiring witness and a
semantic edge whose receipts match the graph, query, episode, relation, and
endpoints. Empty submissions are failures, not malformed certificates.

## Embodied revalidation

All methods receive identical observations, evidence updates, action model, and
deterministic planner. Complete validation, dependency-indexed validation, and
full replanning must never authorize an action after its final support is
withdrawn. The efficiency comparison uses irrelevant withdrawals on retained
plans of length at least 64 and reports paired layout-level intervals.

## Statistical reporting

Stochastic graph policies use seeds 17, 29, and 43. Confidence intervals are
paired over queries or layouts as stated in each table. The certificate mutation
suite is a correlated coverage audit and is not interpreted as independent
Bernoulli attack trials.
