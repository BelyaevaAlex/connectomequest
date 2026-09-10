"""Episode metrics that separate answer quality, proof quality, and cost."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean

from connectomequest.agents.explorer import EpisodeResult
from connectomequest.query import Query


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    query_id: str
    goal_success: bool
    proof_valid: bool
    precision: float
    recall: float
    f1: float
    steps: int
    budget_used: float
    invalid_actions: int
    provenance_completeness: float
    first_proof_reciprocal_rank: float


def episode_metrics(query: Query, result: EpisodeResult) -> EpisodeMetrics:
    predicted = set(result.submitted_answers)
    expected = set(query.answers)
    true_positive = len(predicted & expected)
    precision = true_positive / len(predicted) if predicted else float(not expected)
    recall = true_positive / len(expected) if expected else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    proof_valid = bool(predicted and result.validation and result.validation.evidence_valid)
    provenance_completeness = (
        mean(bool(item.source_record) for item in result.evidence)
        if result.evidence
        else float(not predicted)
    )
    return EpisodeMetrics(
        query_id=query.query_id,
        goal_success=bool(result.validation and result.validation.valid),
        proof_valid=proof_valid,
        precision=precision,
        recall=recall,
        f1=f1,
        steps=result.steps,
        budget_used=result.budget_used,
        invalid_actions=result.invalid_actions,
        provenance_completeness=provenance_completeness,
        first_proof_reciprocal_rank=(
            1.0 / result.first_proof_rank if result.first_proof_rank is not None else 0.0
        ),
    )


def aggregate_metrics(rows: list[EpisodeMetrics]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        "episodes": float(len(rows)),
        "success_at_budget": mean(row.goal_success for row in rows),
        "proof_validity": mean(row.proof_valid for row in rows),
        "precision": mean(row.precision for row in rows),
        "recall": mean(row.recall for row in rows),
        "f1": mean(row.f1 for row in rows),
        "mean_steps": mean(row.steps for row in rows),
        "mean_budget_used": mean(row.budget_used for row in rows),
        "invalid_action_rate": sum(row.invalid_actions for row in rows)
        / max(1, sum(row.steps for row in rows)),
        "provenance_completeness": mean(row.provenance_completeness for row in rows),
        "mean_first_proof_reciprocal_rank": mean(row.first_proof_reciprocal_rank for row in rows),
    }


def metrics_as_dict(row: EpisodeMetrics) -> dict:
    return asdict(row)


def filtered_ranks(ranked_entities: list[int], answers: frozenset[int]) -> list[int]:
    """Ranks of every answer after removing all other true answers."""

    ranks: list[int] = []
    false_entities_seen = 0
    for entity in ranked_entities:
        if entity in answers:
            ranks.append(false_entities_seen + 1)
        else:
            false_entities_seen += 1
    if len(ranks) != len(answers):
        missing = len(answers) - len(ranks)
        raise ValueError(f"ranking omitted {missing} answer entities")
    return ranks


def raw_ranks(ranked_entities: list[int], answers: frozenset[int]) -> list[int]:
    """Unfiltered one-indexed ranks of every answer in a complete ranking."""

    if len(ranked_entities) != len(set(ranked_entities)):
        raise ValueError("ranking contains duplicate entities")
    positions = {entity: index + 1 for index, entity in enumerate(ranked_entities)}
    missing = answers - positions.keys()
    if missing:
        raise ValueError(f"ranking omitted {len(missing)} answer entities")
    return sorted(positions[entity] for entity in answers)


def rank_metrics(ranks: list[int]) -> dict[str, float]:
    """Answer-weighted reciprocal-rank and Hits@K metrics."""

    if not ranks:
        return {
            "mrr": 0.0,
            "hits_at_1": 0.0,
            "hits_at_3": 0.0,
            "hits_at_10": 0.0,
            "hits_at_50": 0.0,
        }
    return {
        "mrr": mean(1.0 / rank for rank in ranks),
        **{
            f"hits_at_{cutoff}": mean(rank <= cutoff for rank in ranks) for cutoff in (1, 3, 10, 50)
        },
    }


def aggregate_frontier_ranks(
    raw: list[int],
    filtered: list[int],
    first_proof: list[int],
    *,
    candidate_counts: list[int],
    relevant_counts: list[int],
) -> dict[str, object]:
    """Aggregate partial-observation frontier ranking metrics.

    Raw and filtered ranks are answer-weighted over all useful middle nodes.
    First-proof ranks are query-weighted and match the active find-one-proof goal.
    """

    if len(candidate_counts) != len(relevant_counts) or len(first_proof) != len(candidate_counts):
        raise ValueError("frontier metric inputs have inconsistent query counts")
    raw_summary = rank_metrics(raw)
    filtered_summary = rank_metrics(filtered)
    first_summary = rank_metrics(first_proof)
    return {
        "queries": float(len(candidate_counts)),
        "relevant_middles": float(len(raw)),
        "mean_candidates": mean(candidate_counts) if candidate_counts else 0.0,
        "mean_relevant_middles": mean(relevant_counts) if relevant_counts else 0.0,
        "raw": raw_summary,
        "filtered": {
            "filtered_mrr": filtered_summary["mrr"],
            **{key: value for key, value in filtered_summary.items() if key != "mrr"},
        },
        "first_proof": first_summary,
    }


def aggregate_ranks(ranks: list[int], *, queries: int) -> dict[str, float]:
    """Aggregate answer-weighted filtered CQA metrics."""

    if not ranks:
        return {
            "queries": float(queries),
            "answers": 0.0,
            "filtered_mrr": 0.0,
            "hits_at_1": 0.0,
            "hits_at_3": 0.0,
            "hits_at_10": 0.0,
            "hits_at_50": 0.0,
        }
    return {
        "queries": float(queries),
        "answers": float(len(ranks)),
        "filtered_mrr": mean(1.0 / rank for rank in ranks),
        **{
            f"hits_at_{cutoff}": mean(rank <= cutoff for rank in ranks) for cutoff in (1, 3, 10, 50)
        },
    }
