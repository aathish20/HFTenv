"""
Data models for the HFT Security Environment.

Defines the action, observation, and state models for interacting
with the HFTSecurityEnv (high-frequency trading simulation).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class HFTOpportunity(BaseModel):
    """Single trading opportunity available in the market."""

    value: int
    latency_sensitivity: str  # "low", "medium", "high"
    signal_strength: float = 0.0
    anomaly_score: float = 0.0


class HFTAction(Action):
    """Action issued by the agent each step.

    Controls three variables:
        - security_level: float in [0, 1]
        - active_nodes: int in [1, 10]
        - selected_indices: list[int] of opportunity indices to execute
    """

    security_level: float
    active_nodes: int
    selected_indices: List[int] = Field(default_factory=list)


class HFTObservation(Observation):
    """Observation returned from the HFT environment each step."""

    prompt: str
    step: int = 0
    latency_ms: float = 1.0
    active_nodes: int = 10
    security_level: float = 0.0
    available_opportunities: int = 10
    selected_indices: List[int] = Field(default_factory=list)
    cumulative_profit: float = 0.0
    cumulative_loss: float = 0.0
    target_profit: float = 1000.0
    node_health: List[float] = Field(default_factory=list)
    node_compromised: List[bool] = Field(default_factory=list)
    system_stress: float = 0.0
    time_of_day: str = "09:00"
    required_avg_net_per_hour: float = 0.0
    time_step: int = 0
    max_steps: int = 10
    opportunities: List[HFTOpportunity] = Field(default_factory=list)
    step_profit: float = 0.0
    step_loss: float = 0.0
    info: Dict[str, Any] = Field(default_factory=dict)


class HFTState(State):
    """Structured state snapshot for the server."""

    episode_id: Optional[str] = None
    step_count: int = 0
    task_id: str = "easy"
    total_steps: int = 10
    total_nodes: int = 10
    available_opportunities: int = 10
    base_latency_ms: float = 1.0
    max_latency_ms: float = 5.0
    cumulative_profit: float = 0.0
    cumulative_loss: float = 0.0
    target_profit: float = 1000.0
    node_health: List[float] = Field(default_factory=list)
    node_compromised: List[bool] = Field(default_factory=list)
    system_stress: float = 0.0
    last_reward: float = 0.0
    last_info: Dict[str, Any] = Field(default_factory=dict)
    raw_state: Dict[str, Any] = Field(default_factory=dict)
