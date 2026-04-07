"""
HFT Security Environment HTTP Client.

Provides the client for connecting to a HFTSecurityEnvironment server over HTTP.
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import HFTAction, HFTObservation, HFTOpportunity, HFTState


class HFTSecurityEnv(EnvClient[HFTAction, HFTObservation, HFTState]):
    """
    HTTP client for the HFT Security Environment.

    Example:
        >>> client = HFTSecurityEnv(base_url="<ENV_HTTP_URL_HERE>")
        >>> result = client.reset()
        >>> print(result.observation.prompt)
        >>>
        >>> result = client.step(HFTAction(
        ...     security_level=0.5,
        ...     active_nodes=8,
        ...     selected_indices=[0, 2, 5],
        ... ))
        >>> print(result.observation.cumulative_profit)
        >>> print(result.reward)

    Example with Docker:
        >>> client = HFTSecurityEnv.from_docker_image("hftenv:latest")
        >>> result = client.reset()
    """

    def _step_payload(self, action: HFTAction) -> Dict:
        """Convert HFTAction to JSON payload for step request."""
        return {
            "security_level": action.security_level,
            "active_nodes": action.active_nodes,
            "selected_indices": list(action.selected_indices),
        }

    def _parse_result(self, payload: Dict) -> StepResult[HFTObservation]:
        """Parse server response into StepResult[HFTObservation]."""
        obs_data = payload.get("observation", {})

        opportunities = [
            HFTOpportunity(
                value=opp.get("value", 0),
                latency_sensitivity=opp.get("latency_sensitivity", "low"),
                signal_strength=float(opp.get("signal_strength", 0.0)),
                anomaly_score=float(opp.get("anomaly_score", 0.0)),
            )
            for opp in obs_data.get("opportunities", [])
            if isinstance(opp, dict)
        ]

        observation = HFTObservation(
            prompt=obs_data.get("prompt", ""),
            step=obs_data.get("step", 0),
            latency_ms=obs_data.get("latency_ms", 1.0),
            active_nodes=obs_data.get("active_nodes", 10),
            security_level=obs_data.get("security_level", 0.0),
            available_opportunities=obs_data.get("available_opportunities", 10),
            selected_indices=list(obs_data.get("selected_indices", [])),
            cumulative_profit=obs_data.get("cumulative_profit", 0.0),
            cumulative_loss=obs_data.get("cumulative_loss", 0.0),
            target_profit=obs_data.get("target_profit", 1000.0),
            node_health=list(obs_data.get("node_health", [])),
            node_compromised=list(obs_data.get("node_compromised", [])),
            system_stress=float(obs_data.get("system_stress", 0.0)),
            time_of_day=obs_data.get("time_of_day", "09:00"),
            required_avg_net_per_hour=float(obs_data.get("required_avg_net_per_hour", 0.0)),
            time_step=int(obs_data.get("time_step", 0)),
            max_steps=int(obs_data.get("max_steps", 10)),
            opportunities=opportunities,
            step_profit=obs_data.get("step_profit", 0.0),
            step_loss=obs_data.get("step_loss", 0.0),
            info=obs_data.get("info", {}),
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> HFTState:
        """Parse server state response into HFTState."""
        return HFTState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", "easy"),
            total_steps=payload.get("total_steps", 10),
            total_nodes=payload.get("total_nodes", 10),
            available_opportunities=payload.get("available_opportunities", 10),
            base_latency_ms=payload.get("base_latency_ms", 1.0),
            max_latency_ms=payload.get("max_latency_ms", 5.0),
            cumulative_profit=payload.get("cumulative_profit", 0.0),
            cumulative_loss=payload.get("cumulative_loss", 0.0),
            target_profit=payload.get("target_profit", 1000.0),
            last_reward=payload.get("last_reward", 0.0),
            last_info=payload.get("last_info", {}),
            raw_state=payload.get("raw_state", {}),
        )
