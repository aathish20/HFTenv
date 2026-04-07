"""Deterministic task graders for HFT Security Environment.

These graders are first-class scoring components used at episode end.
Each task returns a score in [0.0, 1.0] with transparent sub-components.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping


MIN_VALID_SCORE = 0.002
MAX_VALID_SCORE = 0.998


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _clamp_open_unit_interval(x: float) -> float:
    """Clamp score to a strict evaluator-safe open interval."""
    return _clamp(x, MIN_VALID_SCORE, MAX_VALID_SCORE)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _task_weights(task_id: str) -> Dict[str, float]:
    if task_id == "easy":
        return {
            "profit_progress": 0.75,
            "loss_control": 0.15,
            "ops_continuity": 0.10,
        }
    if task_id == "medium":
        return {
            "profit_progress": 0.60,
            "loss_control": 0.15,
            "stability": 0.15,
            "ops_continuity": 0.10,
        }
    if task_id == "hard":
        return {
            "profit_progress": 0.50,
            "loss_control": 0.15,
            "resilience": 0.20,
            "stability": 0.10,
            "ops_continuity": 0.05,
        }
    # very_hard (and fallback)
    return {
        "profit_progress": 0.45,
        "loss_control": 0.15,
        "resilience": 0.20,
        "stability": 0.15,
        "ops_continuity": 0.05,
    }


def _component_scores(
    *,
    target_profit: float,
    cumulative_profit: float,
    cumulative_loss: float,
    total_steps: int,
    episode_stats: Mapping[str, Any],
) -> Dict[str, float]:
    steps = max(int(total_steps), 1)
    net_profit = max(0.0, float(cumulative_profit) - float(cumulative_loss))

    profit_progress = _clamp(_safe_ratio(net_profit, float(target_profit)))

    # Loss control uses a task-agnostic normalization budget.
    # If losses exceed 75% of target profit, this component trends to zero.
    loss_ratio = _safe_ratio(float(cumulative_loss), max(float(target_profit) * 0.75, 1.0))
    loss_control = _clamp(1.0 - loss_ratio)

    total_failed_nodes = float(episode_stats.get("total_failed_nodes", 0.0))
    # If average failed nodes per step approaches 4, stability drops sharply.
    stability = _clamp(1.0 - _safe_ratio(total_failed_nodes, float(steps) * 4.0))

    total_compromises = float(episode_stats.get("total_compromises", 0.0))
    total_malicious_executed = float(episode_stats.get("total_malicious_executed", 0.0))
    compromise_rate = _safe_ratio(total_compromises, float(steps))
    malicious_rate = _safe_ratio(total_malicious_executed, float(steps))
    resilience = _clamp(1.0 - (0.8 * compromise_rate + 0.4 * malicious_rate))

    zero_trade_steps = float(episode_stats.get("zero_trade_steps", 0.0))
    ops_continuity = _clamp(1.0 - _safe_ratio(zero_trade_steps, float(steps)))

    return {
        "profit_progress": float(profit_progress),
        "loss_control": float(loss_control),
        "stability": float(stability),
        "resilience": float(resilience),
        "ops_continuity": float(ops_continuity),
    }


def _grade_episode_internal(
    *,
    task_id: str,
    target_profit: float,
    cumulative_profit: float,
    cumulative_loss: float,
    total_steps: int,
    episode_stats: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return deterministic task grade in [0.0, 1.0] with component breakdown."""
    components = _component_scores(
        target_profit=target_profit,
        cumulative_profit=cumulative_profit,
        cumulative_loss=cumulative_loss,
        total_steps=total_steps,
        episode_stats=episode_stats,
    )
    weights = _task_weights(task_id)

    weighted_sum = 0.0
    for key, weight in weights.items():
        weighted_sum += float(components.get(key, 0.0)) * float(weight)
    # Keep final score strictly within (0, 1), never exactly 0.0 or 1.0.
    score = _clamp_open_unit_interval(weighted_sum)

    return {
        "task_id": task_id,
        "score": float(score),
        "weights": {k: float(v) for k, v in weights.items()},
        "components": components,
    }


def grade_easy(
    *,
    target_profit: float,
    cumulative_profit: float,
    cumulative_loss: float,
    total_steps: int,
    episode_stats: Mapping[str, Any],
) -> Dict[str, Any]:
    return _grade_episode_internal(
        task_id="easy",
        target_profit=target_profit,
        cumulative_profit=cumulative_profit,
        cumulative_loss=cumulative_loss,
        total_steps=total_steps,
        episode_stats=episode_stats,
    )


def grade_medium(
    *,
    target_profit: float,
    cumulative_profit: float,
    cumulative_loss: float,
    total_steps: int,
    episode_stats: Mapping[str, Any],
) -> Dict[str, Any]:
    return _grade_episode_internal(
        task_id="medium",
        target_profit=target_profit,
        cumulative_profit=cumulative_profit,
        cumulative_loss=cumulative_loss,
        total_steps=total_steps,
        episode_stats=episode_stats,
    )


def grade_hard(
    *,
    target_profit: float,
    cumulative_profit: float,
    cumulative_loss: float,
    total_steps: int,
    episode_stats: Mapping[str, Any],
) -> Dict[str, Any]:
    return _grade_episode_internal(
        task_id="hard",
        target_profit=target_profit,
        cumulative_profit=cumulative_profit,
        cumulative_loss=cumulative_loss,
        total_steps=total_steps,
        episode_stats=episode_stats,
    )


def grade_very_hard(
    *,
    target_profit: float,
    cumulative_profit: float,
    cumulative_loss: float,
    total_steps: int,
    episode_stats: Mapping[str, Any],
) -> Dict[str, Any]:
    return _grade_episode_internal(
        task_id="very_hard",
        target_profit=target_profit,
        cumulative_profit=cumulative_profit,
        cumulative_loss=cumulative_loss,
        total_steps=total_steps,
        episode_stats=episode_stats,
    )


TASK_GRADERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
    "very_hard": grade_very_hard,
}


def list_graded_tasks() -> list[str]:
    return list(TASK_GRADERS.keys())


def grade_episode(
    *,
    task_id: str,
    target_profit: float,
    cumulative_profit: float,
    cumulative_loss: float,
    total_steps: int,
    episode_stats: Mapping[str, Any],
) -> Dict[str, Any]:
    grader = TASK_GRADERS.get(task_id)
    if grader is None:
        # Fallback keeps behavior deterministic for unknown task ids.
        return _grade_episode_internal(
            task_id=task_id,
            target_profit=target_profit,
            cumulative_profit=cumulative_profit,
            cumulative_loss=cumulative_loss,
            total_steps=total_steps,
            episode_stats=episode_stats,
        )
    return grader(
        target_profit=target_profit,
        cumulative_profit=cumulative_profit,
        cumulative_loss=cumulative_loss,
        total_steps=total_steps,
        episode_stats=episode_stats,
    )
