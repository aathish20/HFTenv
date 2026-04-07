"""Reward provider utilities for HFT Security Environment."""

from __future__ import annotations

from typing import Dict, List, Protocol

try:
    from models import HFTAction, HFTObservation
except ImportError:
    from .models import HFTAction, HFTObservation


class RewardProvider(Protocol):
    """Interface for computing auxiliary reward signals."""

    def reset(self) -> None: ...

    def compute(
        self, *, action: HFTAction, observation: HFTObservation
    ) -> Dict[str, float]: ...


def build_reward_providers(task_id: str) -> List[RewardProvider]:
    """Instantiate reward providers appropriate for the given task."""
    providers: List[RewardProvider] = []
    providers.append(_HFTRewardProvider())
    return providers


class _HFTRewardProvider:
    """Computes four auxiliary reward signals for the HFT environment."""

    SIGNAL_MAP = {
        "profit_ratio": "hft.profit_ratio",
        "risk_management": "hft.risk_management",
        "cost_efficiency": "hft.cost_efficiency",
        "loss_control": "hft.loss_control",
    }

    def __init__(self) -> None:
        self._step_count = 0

    def reset(self) -> None:
        self._step_count = 0

    def compute(
        self, *, action: HFTAction, observation: HFTObservation
    ) -> Dict[str, float]:
        self._step_count += 1

        # Profit score: normalized net profit progress toward target
        # profit_score = clamp(net_profit / target_profit, 0, 1)
        net_profit = float(observation.cumulative_profit - observation.cumulative_loss)
        if observation.target_profit > 0:
            profit_ratio = min(1.0, max(0.0, net_profit / float(observation.target_profit)))
        else:
            profit_ratio = 0.0

        # Risk management: reward balanced security vs selection aggressiveness
        avail = max(1, observation.available_opportunities)
        selected_frac = min(1.0, max(0.0, len(getattr(action, "selected_indices", []) or []) / avail))
        risk_score = action.security_level * 0.6 + (1.0 - selected_frac) * 0.4

        # Cost efficiency: penalize excessive spending
        node_cost = action.active_nodes * 5.0
        # Keep consistent with environment cost model
        security_cost = 15.0 * (action.security_level ** 2)
        total_cost = node_cost + security_cost
        cost_efficiency = max(0.0, 1.0 - total_cost / 100.0)

        # Loss control: penalize cumulative loss relative to target
        loss_control = max(
            0.0,
            1.0 - observation.cumulative_loss / max(1.0, observation.target_profit),
        )

        return {
            self.SIGNAL_MAP["profit_ratio"]: profit_ratio,
            self.SIGNAL_MAP["risk_management"]: risk_score,
            self.SIGNAL_MAP["cost_efficiency"]: cost_efficiency,
            self.SIGNAL_MAP["loss_control"]: loss_control,
        }


__all__ = [
    "RewardProvider",
    "build_reward_providers",
]
