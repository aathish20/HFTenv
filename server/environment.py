"""Server implementation for the HFT Security Environment."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from hftenv.graders import grade_episode, list_graded_tasks
    from hftenv.models import HFTAction, HFTObservation, HFTOpportunity, HFTState
    from hftenv.tasks import (
        LATENCY_PENALTY_MAP,
        TaskConfig,
        generate_opportunities,
        get_task,
    )
    from hftenv.rewards import RewardProvider, build_reward_providers
except ImportError:
    from graders import grade_episode, list_graded_tasks  # type: ignore[no-redef]
    from models import HFTAction, HFTObservation, HFTOpportunity, HFTState  # type: ignore[no-redef]
    from tasks import (  # type: ignore[no-redef]
        LATENCY_PENALTY_MAP,
        TaskConfig,
        generate_opportunities,
        get_task,
    )
    from rewards import RewardProvider, build_reward_providers  # type: ignore[no-redef]

# ── Constants ────────────────────────────────────────────────────────────────
# TOTAL_STEPS is task-dependent (see TaskConfig.max_steps). Keep a default for metadata.
TOTAL_STEPS = 10
TOTAL_NODES = 10
AVAILABLE_OPPORTUNITIES = 10
BASE_LATENCY_MS = 1.0
MAX_LATENCY_MS = 5.0


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class HFTSecurityEnvironment(Environment):
    """HFT Security trading simulation behind the OpenEnv Environment API.

    The agent must balance speed (latency), capacity (nodes), security
    (filtering & screening), and cost to maximize cumulative profit
    over a fixed number of steps while minimizing losses from latency,
    filtering, and bad trades.
    """

    def __init__(self, task_id: str = "easy") -> None:
        super().__init__()
        self._task: TaskConfig = get_task(task_id)
        self._state = HFTState(
            task_id=task_id,
            total_steps=self._task.max_steps,
            total_nodes=TOTAL_NODES,
            available_opportunities=AVAILABLE_OPPORTUNITIES,
            base_latency_ms=BASE_LATENCY_MS,
            max_latency_ms=MAX_LATENCY_MS,
            target_profit=self._task.target_profit,
            node_health=[1.0 for _ in range(TOTAL_NODES)],
            node_compromised=[False for _ in range(TOTAL_NODES)],
            system_stress=0.0,
        )
        self._current_opportunities: List[Dict[str, Any]] = []
        self._reward_providers: List[RewardProvider] = build_reward_providers(task_id)
        self._last_reward_signals: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> HFTObservation:
        task_id = kwargs.get("task_id", self._state.task_id)
        self._task = get_task(task_id)

        # Deterministic episode seed: if provided, use it; else derive from task base_seed.
        episode_seed = int(seed) if seed is not None else int(self._task.base_seed)

        self._state = HFTState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            total_steps=self._task.max_steps,
            total_nodes=TOTAL_NODES,
            available_opportunities=AVAILABLE_OPPORTUNITIES,
            base_latency_ms=BASE_LATENCY_MS,
            max_latency_ms=MAX_LATENCY_MS,
            cumulative_profit=0.0,
            cumulative_loss=0.0,
            target_profit=self._task.target_profit,
            node_health=[1.0 for _ in range(TOTAL_NODES)],
            node_compromised=[False for _ in range(TOTAL_NODES)],
            system_stress=0.0,
            last_reward=0.0,
            last_info={},
            raw_state={
                "episode_seed": episode_seed,
                "episode_stats": {
                    "steps": 0,
                    "zero_trade_steps": 0,
                    "total_executed_trades": 0,
                    "total_failed_nodes": 0,
                    "total_compromises": 0,
                    "total_malicious_executed": 0,
                    "total_latency_attack_events": 0,
                    "total_latency_attack_steps": 0,
                },
            },
        )

        for provider in self._reward_providers:
            provider.reset()

        self._current_opportunities = generate_opportunities(
            self._task, step=0, count=AVAILABLE_OPPORTUNITIES
        )
        self._last_reward_signals = {}

        obs = self._build_observation(step_profit=0.0, step_loss=0.0)
        obs.reward = 0.0
        obs.done = False
        return obs

    def step(self, action: HFTAction) -> HFTObservation:  # type: ignore[override]
        if not isinstance(action, HFTAction):
            raise TypeError(f"Expected HFTAction, received {type(action)!r}")

        # ── Clamp action values ──────────────────────────────────────────
        security_level = max(0.0, min(1.0, action.security_level))
        # active_nodes must be 1..10 in the new spec
        active_nodes = max(1, min(TOTAL_NODES, action.active_nodes))
        raw_indices = list(getattr(action, "selected_indices", []) or [])

        current_step = self._state.step_count
        max_steps = int(self._state.total_steps)

        # Deterministic per-episode seed
        episode_seed = int(self._state.raw_state.get("episode_seed", self._task.base_seed))

        # Persistent failure bias (recovery can reduce it over time)
        self._state.raw_state.setdefault("failure_bias", 1.0)

        # Persistent attack timers/state
        latency_attack_steps_left = int(self._state.raw_state.get("latency_attack_steps_left", 0))
        latency_attack_spike_left = float(self._state.raw_state.get("latency_attack_spike_left", 0.0))
        attack_cooldown_left = int(self._state.raw_state.get("attack_cooldown_left", 0))
        compromised_steps_left: List[int] = list(
            self._state.raw_state.get("compromised_steps_left", [0 for _ in range(TOTAL_NODES)])
        )
        episode_stats: Dict[str, Any] = dict(
            self._state.raw_state.get(
                "episode_stats",
                {
                    "steps": 0,
                    "zero_trade_steps": 0,
                    "total_executed_trades": 0,
                    "total_failed_nodes": 0,
                    "total_compromises": 0,
                    "total_malicious_executed": 0,
                    "total_latency_attack_events": 0,
                    "total_latency_attack_steps": 0,
                },
            )
        )

        # ── 1) Generate Opportunities (per tick) ─────────────────────────
        # Always regenerate opportunities each step for a realistic market tick.
        self._current_opportunities = generate_opportunities(
            self._task, step=current_step, count=AVAILABLE_OPPORTUNITIES
        )

        # ── Latency model (fixed): latency is computed per-step, not accumulated ──
        # Smooth latency scaling with security (avoid overly steep linear penalty).
        # Other factors only affect latency within the current step (e.g., active attack).
        latency_ms = BASE_LATENCY_MS + 12.0 * (security_level ** 1.35)

        # High security (>0.7) adds extra overhead (5-15ms) deterministically
        security_latency_overhead = 0.0
        if security_level > 0.7:
            rng_sec_lat = random.Random(episode_seed + current_step * 1000 + 150)
            security_latency_overhead = rng_sec_lat.uniform(5.0, 15.0)
            latency_ms += security_latency_overhead

        # Persistent latency attack effect (task-tuned duration, decays per step)
        if latency_attack_steps_left > 0:
            latency_ms += max(0.0, latency_attack_spike_left)
            latency_attack_spike_left = max(0.0, latency_attack_spike_left - 10.0)
            latency_attack_steps_left -= 1

        # ── 2) Security-based filtering (soft) ───────────────────────────
        # Security does NOT remove opportunities from the observation.
        # Instead, it determines which indices are considered "allowed" to execute.
        # Higher security => stricter anomaly threshold.
        # Reduce aggressiveness: keep more trades executable even at high security.
        anomaly_threshold = 0.90 - 0.45 * security_level  # sec=0 -> 0.90, sec=1 -> 0.45
        allowed_indices = {
            i
            for i, opp in enumerate(self._current_opportunities)
            if float(opp.get("anomaly_score", 0.0)) <= anomaly_threshold
        }
        filtered_out = int(AVAILABLE_OPPORTUNITIES - len(allowed_indices))

        # ── 3) Apply Cyber Attacks (persistent + escalates over time) ────
        # Attack escalation over time (task-scaled)
        progress = (current_step + 1) / max(1.0, float(max_steps))
        base_attack = float(getattr(self._task, "attack_base_prob", 0.10))
        mid_bonus = float(getattr(self._task, "attack_progress_mid_bonus", 0.20))
        late_bonus = float(getattr(self._task, "attack_progress_late_bonus", 0.40))
        low_security_bonus = float(getattr(self._task, "low_security_attack_bonus", 0.30))
        if progress < 0.30:
            attack_p = base_attack
        elif progress < 0.70:
            attack_p = min(1.0, base_attack + mid_bonus)
        else:
            attack_p = min(1.0, base_attack + late_bonus)

        # Increase attack pressure late game
        if progress > 0.5:
            attack_p = min(1.0, attack_p + 0.2)

        # Low security increases attack probability and fake/compromise likelihood
        if security_level < 0.4:
            attack_p = _clamp(attack_p + low_security_bonus, 0.0, 1.0)
        attack_p = _clamp(attack_p * float(getattr(self._task, "attack_scale", 1.0)), 0.0, 1.0)

        rng_attack = random.Random(episode_seed + current_step * 1000 + 500)
        latency_attack_triggered = False
        node_compromise_triggered = False
        fake_signals_triggered = False
        latency_spike = 0.0
        compromised_node_idx: Optional[int] = None

        if attack_cooldown_left > 0:
            attack_cooldown_left -= 1

        if attack_cooldown_left <= 0 and rng_attack.random() < attack_p:
            # Bias attack type based on security
            if security_level < 0.4:
                weights = [0.25, 0.40, 0.35]  # latency, compromise, fake
            else:
                weights = [0.40, 0.30, 0.30]
            attack_type = rng_attack.choices(["latency", "compromise", "fake"], weights=weights, k=1)[0]
            attack_cooldown_left = int(getattr(self._task, "attack_cooldown_steps", 3))

            if attack_type == "latency":
                latency_attack_triggered = True
                spike_lo, spike_hi = list(
                    getattr(self._task, "latency_attack_spike_range", [50.0, 120.0])
                )[:2]
                dur_lo, dur_hi = list(
                    getattr(self._task, "latency_attack_duration_range", [2, 4])
                )[:2]
                latency_spike = rng_attack.uniform(float(spike_lo), float(spike_hi))
                latency_attack_steps_left = rng_attack.randint(int(dur_lo), int(dur_hi))
                latency_attack_spike_left = latency_spike
            elif attack_type == "compromise":
                node_compromise_triggered = True
                compromised_node_idx = rng_attack.randrange(0, TOTAL_NODES)
                self._state.node_compromised[compromised_node_idx] = True
                comp_lo, comp_hi = list(
                    getattr(self._task, "compromise_duration_range", [4, 6])
                )[:2]
                compromised_steps_left[compromised_node_idx] = rng_attack.randint(
                    int(comp_lo), int(comp_hi)
                )
            else:
                # Fake signals are task-scaled and can also occur independently.
                fake_signals_triggered = True
                fake_lo, fake_hi = list(
                    getattr(self._task, "fake_signal_inject_range", [1, 2])
                )[:2]
                inject_n = rng_attack.randint(int(fake_lo), int(fake_hi))
                for _ in range(inject_n):
                    self._current_opportunities.append(
                        {
                            "value": int(rng_attack.randint(300, 500)),
                            "risk": 0.3,
                            "latency_sensitivity": rng_attack.choice(["low", "medium", "high"]),
                            "latency_sensitivity_num": float(rng_attack.random()),
                            "is_malicious": True,
                            "signal_strength": float(rng_attack.uniform(0.55, 0.85)),
                            "anomaly_score": float(rng_attack.uniform(0.55, 0.90)),
                        }
                    )

        # Task-scaled fake signals (can happen even without a chosen attack)
        rng_fake = random.Random(episode_seed + current_step * 1000 + 505)
        if rng_fake.random() < float(getattr(self._task, "fake_signal_rate", 0.10)):
            fake_signals_triggered = True
            fake_lo, fake_hi = list(
                getattr(self._task, "fake_signal_inject_range", [1, 2])
            )[:2]
            inject_n = rng_fake.randint(int(fake_lo), int(fake_hi))
            for _ in range(inject_n):
                self._current_opportunities.append(
                    {
                        "value": int(rng_fake.randint(300, 500)),
                        "risk": 0.3,
                        "latency_sensitivity": rng_fake.choice(["low", "medium", "high"]),
                        "latency_sensitivity_num": float(rng_fake.random()),
                        "is_malicious": True,
                        "signal_strength": float(rng_fake.uniform(0.55, 0.85)),
                        "anomaly_score": float(rng_fake.uniform(0.55, 0.90)),
                    }
                )

        # Decrement compromise timers; clear compromised flag when timer expires
        for i in range(TOTAL_NODES):
            if compromised_steps_left[i] > 0:
                compromised_steps_left[i] -= 1
                if compromised_steps_left[i] <= 0:
                    self._state.node_compromised[i] = False

        # ── SELECT OPPORTUNITIES (by indices) ───────────────────────────
        # Agent provides indices into the *current* opportunity list.
        # We enforce bounds, uniqueness, and security-based filtering.
        seen: set[int] = set()
        bounded_unique: List[int] = []
        for idx in raw_indices:
            try:
                i = int(idx)
            except (TypeError, ValueError):
                continue
            if i < 0 or i >= len(self._current_opportunities):
                continue
            if i in seen:
                continue
            seen.add(i)
            bounded_unique.append(i)

        requested_indices = bounded_unique
        blocked_by_security = [i for i in requested_indices if i not in allowed_indices]
        selected_indices = [i for i in requested_indices if i in allowed_indices]
        selected = [self._current_opportunities[i] for i in selected_indices]

        # Probabilistic malicious execution: even if an index passes the threshold,
        # suspicious trades can still slip through at low security.
        # This makes security meaningful early without hard-blocking everything.
        rng_slip = random.Random(episode_seed + current_step * 1000 + 575)
        executed_malicious_slip = 0
        slip_multiplier = float(getattr(self._task, "slip_multiplier", 0.6))
        if selected:
            slip_selected: List[Dict[str, Any]] = []
            slip_selected_indices: List[int] = []
            for i, opp in zip(selected_indices, selected):
                anomaly = float(opp.get("anomaly_score", 0.0))
                # Only apply slip logic to suspicious trades.
                if anomaly <= 0.6:
                    slip_selected.append(opp)
                    slip_selected_indices.append(i)
                    continue

                # Slip probability increases with anomaly and decreases with security.
                # Starts earlier (anomaly>0.6) to make security matter early.
                base_risk = (anomaly - 0.6) / 0.4
                base_risk = _clamp(base_risk, 0.0, 1.0)
                slip_p = base_risk * (0.6 * (1.0 - security_level)) * slip_multiplier
                if rng_slip.random() < slip_p:
                    executed_malicious_slip += 1
                    slip_selected.append(opp)
                    slip_selected_indices.append(i)
                else:
                    blocked_by_security.append(i)

            selected = slip_selected
            selected_indices = slip_selected_indices

        # ── 4) Node Failure Logic (improved) ─────────────────────────────
        # Node health degrades over time; compromised nodes are less reliable.
        working_nodes = 0
        failed_nodes = 0
        selected_node_indices = list(range(active_nodes))
        node_failure_multiplier = float(getattr(self._task, "node_failure_multiplier", 1.0))
        for node_idx in selected_node_indices:
            health = float(self._state.node_health[node_idx])
            compromised = bool(self._state.node_compromised[node_idx])

            # failure probability per spec
            fail_prob = (
                0.05
                + float(self._state.system_stress) * 0.3
                + (0.4 if compromised else 0.0)
                - security_level * 0.2
            )
            # Persistent failure bias (reduced by recovery when security is high)
            fail_prob *= float(self._state.raw_state.get("failure_bias", 1.0))
            fail_prob *= node_failure_multiplier
            fail_prob = _clamp(fail_prob, 0.0, 1.0)

            rng_node = random.Random(episode_seed + current_step * 1000 + 600 + node_idx)
            if rng_node.random() < fail_prob:
                failed_nodes += 1
                # degrade health more on failure
                self._state.node_health[node_idx] = max(0.0, health - 0.05)
            else:
                working_nodes += 1
                # slight wear even when working
                self._state.node_health[node_idx] = max(0.0, health - 0.01)

        # 5) Execute Trades
        # Capacity constraint: executed trades limited by working nodes.
        executed_trades = min(len(selected), working_nodes)
        executed = selected[:executed_trades] if executed_trades > 0 else []

        # Node failures must matter (reduce execution quality)
        failure_ratio = 0.0
        if active_nodes > 0:
            failure_ratio = float(failed_nodes) / float(active_nodes)

        missed_trades = max(0, len(selected) - executed_trades)

        # ── 6) Profit Calculation ───────────────────────────────────────
        base_profit = float(sum(float(opp["value"]) for opp in selected)) if selected else 0.0

        # latency impact (reduced per spec)
        latency_penalty = min(latency_ms / 150.0, 0.5)

        # Security reduces anomaly effect (screening mitigates adversarial impact)
        anomaly_mitigation = 0.15 + 0.65 * security_level  # sec=0 -> 0.15, sec=1 -> 0.80
        profit = 0.0
        for exec_i, opp in enumerate(executed):
            anomaly = float(opp.get("anomaly_score", 0.0))
            effective_anomaly = _clamp(anomaly * (1.0 - anomaly_mitigation), 0.0, 1.0)
            if anomaly < 0.2:
                effective_anomaly *= 0.5
            # Add small deterministic noise to profits (prevents perfect deterministic policy)
            opp_idx = int(selected_indices[exec_i]) if exec_i < len(selected_indices) else 0
            rng_profit = random.Random(episode_seed + current_step * 1000 + 800 + opp_idx)
            noise = rng_profit.uniform(-0.05, 0.05)
            trade_profit = (
                float(opp["value"])
                * (1.0 - latency_penalty)
                * (1.0 - effective_anomaly)
                * (1.0 + noise)
            )
            # Reduce profit scaling slightly
            trade_profit *= 0.85
            profit += trade_profit

        # ── 7) Loss Conditions ─────────────────────────────────────────-
        malicious_loss = 0.0
        malicious_executed = 0
        security_score_penalty = 0.0
        malicious_penalty_multiplier = float(
            getattr(self._task, "malicious_penalty_multiplier", 2.0)
        )
        for opp in executed:
            if float(opp.get("anomaly_score", 0.0)) > 0.7:
                malicious_executed += 1
                # Increase malicious penalty (task-scaled)
                malicious_loss += float(opp["value"]) * malicious_penalty_multiplier
                security_score_penalty += 0.3

        missed_trade_loss = 0.03 * float(sum(float(opp["value"]) for opp in selected[executed_trades:])) if missed_trades > 0 else 0.0

        # ── COST MODEL (security cost increases significantly at high security) ──
        node_cost = active_nodes * 5.0
        security_cost = 15.0 * (security_level ** 2)
        # Remove trust_cost (selection itself is free; security already costs)
        total_cost = node_cost + security_cost

        # ── 8) Update System State ─────────────────────────────────────-
        # Stress dynamics: stress can build from bad outcomes, but can also recover
        # when the agent invests in security.
        trade_volume = executed_trades / max(1.0, float(AVAILABLE_OPPORTUNITIES))
        stress_delta = (
            0.04 * (active_nodes / float(TOTAL_NODES))
            + 0.05 * trade_volume
            + 0.06 * (1.0 - security_level)
        )
        stress_delta *= float(getattr(self._task, "stress_growth_multiplier", 1.0))
        # Overuse penalty
        if active_nodes > len(selected_indices) + 2:
            stress_delta += 0.05
        # Attacks increase stress
        if latency_attack_triggered:
            stress_delta += 0.05
        if node_compromise_triggered:
            stress_delta += 0.05

        # Additional stress increase based on bad decisions/outcomes (slower growth)
        stress_increase = (
            0.02 * float(failed_nodes)
            + 0.05 * float(malicious_executed)
            + 0.01 * float(len(selected_indices))
        )

        self._state.system_stress = _clamp(
            float(self._state.system_stress) + stress_delta + stress_increase - 0.01,
            0.0,
            1.0,
        )

        # Recovery effect: higher security heals the system (scaled)
        if security_level > 0.3:
            recovery = float(getattr(self._task, "stress_recovery_base", 0.05)) + float(
                getattr(self._task, "stress_recovery_security_scale", 0.10)
            ) * float(security_level)
            self._state.system_stress = max(0.0, float(self._state.system_stress) - recovery)

            # also reduce failure probability indirectly via a persistent bias
            failure_bias = float(self._state.raw_state.get("failure_bias", 1.0))
            failure_bias *= float(getattr(self._task, "failure_bias_recovery", 0.9))
            self._state.raw_state["failure_bias"] = _clamp(failure_bias, 0.5, 1.5)

        # Persist attack timers (latency itself is not accumulated across steps)
        self._state.raw_state["latency_attack_steps_left"] = int(latency_attack_steps_left)
        self._state.raw_state["latency_attack_spike_left"] = float(latency_attack_spike_left)
        self._state.raw_state["attack_cooldown_left"] = int(attack_cooldown_left)
        self._state.raw_state["compromised_steps_left"] = list(compromised_steps_left)

        # Force non-idle behavior
        idle_cost = 0.0
        if len(selected_indices) == 0:
            idle_cost = 20.0

        # Penalty for too few nodes (<2)
        few_nodes_penalty = 0.0
        if active_nodes < 2:
            few_nodes_penalty = 10.0

        # Step accounting
        step_profit = profit - total_cost - malicious_loss
        # Step accounting (balanced penalties + cap)
        # Compute penalties first, then cap total penalty to avoid unrecoverable spirals.
        stress_penalty = float(self._state.system_stress) * 80.0
        collapse_penalty = 100.0 if float(self._state.system_stress) >= 1.0 else 0.0

        # Penalize system instability (only excess failures)
        instability_penalty = 0.0
        if failed_nodes >= 5:
            instability_penalty = float(failed_nodes - 4) * 20.0

        # Node compromise penalty
        compromise_penalty = 100.0 if node_compromise_triggered else 0.0

        # Penalize low security under risk (reduced)
        high_risk_count = sum(
            1 for o in self._current_opportunities if float(o.get("anomaly_score", 0.0)) > 0.7
        )
        low_security_risk_penalty = 60.0 if (high_risk_count >= 3 and security_level < 0.2) else 0.0

        # Force proactive security: punish reckless low-security selection of high-anomaly trades
        reckless_penalty = 0.0
        if security_level < 0.1 and any(
            float(opp.get("anomaly_score", 0.0)) > 0.7 for opp in selected
        ):
            reckless_penalty = 100.0

        total_penalty = (
            stress_penalty
            + collapse_penalty
            + instability_penalty
            + compromise_penalty
            + low_security_risk_penalty
            + reckless_penalty
            + malicious_loss
        )
        penalty_cap = 0.6 * float(base_profit) if base_profit > 0 else 200.0
        total_penalty = min(float(total_penalty), float(penalty_cap))

        step_profit = profit - total_cost - total_penalty

        # Failed nodes reduce execution quality (multiplicative)
        step_profit *= max(0.0, 1.0 - 0.5 * float(failure_ratio))

        # (replaced by continuous stress penalty + multiplicative failure quality + instability penalty)
        step_loss = malicious_loss + missed_trade_loss + idle_cost + few_nodes_penalty

        # Over-selection penalty (selecting more than capacity)
        if len(selected) > working_nodes:
            excess = selected[working_nodes:]
            overflow_loss = 0.02 * float(sum(float(opp["value"]) for opp in excess))
            step_loss += overflow_loss

        self._state.cumulative_profit += step_profit
        self._state.cumulative_loss += step_loss
        self._state.step_count += 1
        self._state.last_reward = step_profit
        episode_stats["steps"] = int(episode_stats.get("steps", 0)) + 1
        episode_stats["total_executed_trades"] = int(
            episode_stats.get("total_executed_trades", 0)
        ) + int(executed_trades)
        episode_stats["zero_trade_steps"] = int(
            episode_stats.get("zero_trade_steps", 0)
        ) + int(1 if executed_trades == 0 else 0)
        episode_stats["total_failed_nodes"] = int(
            episode_stats.get("total_failed_nodes", 0)
        ) + int(failed_nodes)
        episode_stats["total_compromises"] = int(
            episode_stats.get("total_compromises", 0)
        ) + int(1 if node_compromise_triggered else 0)
        episode_stats["total_malicious_executed"] = int(
            episode_stats.get("total_malicious_executed", 0)
        ) + int(malicious_executed)
        episode_stats["total_latency_attack_events"] = int(
            episode_stats.get("total_latency_attack_events", 0)
        ) + int(1 if latency_attack_triggered else 0)
        episode_stats["total_latency_attack_steps"] = int(
            episode_stats.get("total_latency_attack_steps", 0)
        ) + int(1 if (latency_attack_triggered or latency_attack_steps_left > 0) else 0)

        # ── TERMINATION ──────────────────────────────────────────────────
        # Always run the full episode length. The environment dynamics (attacks,
        # node health/compromise, stress) evolve over time and must be experienced.
        done = self._state.step_count >= max_steps

        # ── Build observation ─────────────────────────────────────────────
        obs = self._build_observation(
            step_profit=step_profit,
            step_loss=step_loss,
            latency_ms=latency_ms,
            active_nodes=active_nodes,
            security_level=security_level,
            selected_indices=selected_indices,
        )
        obs.done = done

        # ── Reward Function (step-level partial reward) ───────────────────
        # Training reward must reflect *this step's* decision quality.
        # Keep diminishing returns once the agent is beyond the target.
        net_profit = float(self._state.cumulative_profit - self._state.cumulative_loss)
        if self._task.target_profit > 0 and net_profit > self._task.target_profit:
            # Compress upside beyond target so late-episode tail doesn't dominate.
            # Example: 2x target -> ~1.5x effective.
            over_ratio = (net_profit / self._task.target_profit) - 1.0
            net_profit = self._task.target_profit * (1.0 + 0.5 * over_ratio)

        reward = float(step_profit - step_loss)
        reward -= 0.1 * float(self._state.system_stress)
        reward -= 0.05 * float(latency_ms)

        # Penalty for very high security (>0.7): operational overhead / false positives
        if security_level > 0.7:
            reward -= 2.0 * float((security_level - 0.7) / 0.3)

        # Penalty for zero trades executed (stronger than idle_cost alone)
        if executed_trades == 0:
            reward -= 5.0

        # Penalty for too few nodes (<2)
        if active_nodes < 2:
            reward -= 2.0

        if malicious_executed > 0:
            reward -= float(malicious_loss)
        if node_compromise_triggered:
            reward -= 1.0
        reward -= 0.2 * float(failed_nodes)

        # Progress delta (not cumulative)
        reward += 0.3 * float(step_profit - step_loss)

        # Terminal shaping
        if done:
            if self._state.cumulative_profit >= self._task.target_profit:
                reward += float(getattr(self._task, "terminal_bonus_success", 5.0))
            else:
                reward += float(getattr(self._task, "terminal_bonus_failure", -5.0))

        obs.reward = reward
        if done:
            net_profit = max(
                0.0, float(self._state.cumulative_profit - self._state.cumulative_loss)
            )
            profit_progress_score = (
                _clamp(net_profit / float(self._task.target_profit), 0.0, 1.0)
                if float(self._task.target_profit) > 0.0
                else 0.0
            )
            task_grade = grade_episode(
                task_id=self._task.task_id,
                target_profit=float(self._task.target_profit),
                cumulative_profit=float(self._state.cumulative_profit),
                cumulative_loss=float(self._state.cumulative_loss),
                total_steps=int(self._state.total_steps),
                episode_stats=episode_stats,
            )
            obs.info["final_score"] = float(task_grade["score"])
            obs.info["task_grade"] = task_grade
            obs.info["grader_task_id"] = self._task.task_id
            obs.info["graded_tasks"] = list_graded_tasks()
            obs.info["profit_progress_score"] = float(profit_progress_score)
            obs.info["raw_final_reward"] = float(reward)
            obs.info["net_profit"] = float(net_profit)
            obs.info["cumulative_profit"] = self._state.cumulative_profit
            obs.info["cumulative_loss"] = self._state.cumulative_loss
            obs.info["target_profit"] = self._task.target_profit
            obs.info["episode_stats"] = episode_stats
            obs.info["success_threshold"] = float(
                getattr(self._task, "reward_target_threshold", 0.60)
            )

        # Reward signals
        reward_signals = self._compute_reward_signals(action=action, observation=obs)
        if reward_signals:
            obs.info.setdefault("reward_signals", {}).update(reward_signals)
            obs.metadata.setdefault("reward_signals", {}).update(reward_signals)
        self._last_reward_signals = reward_signals

        self._state.last_info = {
            "step_profit": step_profit,
            "step_loss": step_loss,
            "malicious_loss": malicious_loss,
            "security_score_penalty": security_score_penalty,
            "missed_trade_loss": missed_trade_loss,
            "idle_cost": idle_cost,
            "stress_penalty": float(stress_penalty),
            "collapse_penalty": float(collapse_penalty),
            "failure_ratio": float(failure_ratio),
            "instability_penalty": float(instability_penalty),
            "compromise_penalty": float(compromise_penalty),
            "low_security_risk_penalty": float(low_security_risk_penalty),
            "total_cost": total_cost,
            "latency_ms": latency_ms,
            "base_profit": base_profit,
            "profit": profit,
            "executed_trades": executed_trades,
            "malicious_executed": malicious_executed,
            "anomaly_threshold": float(anomaly_threshold),
            "filtered_out": int(filtered_out),
            "blocked_by_security": list(blocked_by_security),
            "selected_indices": list(selected_indices),
            "executed_malicious_slip": int(executed_malicious_slip),
            "working_nodes": working_nodes,
            "failed_nodes": failed_nodes,
            "latency_attack": float(latency_spike),
            "latency_attack_steps_left": int(latency_attack_steps_left),
            "attack_cooldown_left": int(attack_cooldown_left),
            "node_compromise": bool(node_compromise_triggered),
            "compromised_node_idx": compromised_node_idx,
            "fake_signals": bool(fake_signals_triggered),
            "system_stress": float(self._state.system_stress),
            "reward": float(obs.reward),
            "episode_stats": episode_stats,
            "reward_components": {
                "step_profit": float(step_profit),
                "step_loss": float(step_loss),
                "system_stress": float(self._state.system_stress),
                "latency_ms": float(latency_ms),
                "malicious_loss": float(malicious_loss),
                "malicious_executed": float(malicious_executed),
                "failed_nodes": float(failed_nodes),
                "node_compromise": float(1.0 if node_compromise_triggered else 0.0),
            },
            **({"reward_signals": reward_signals} if reward_signals else {}),
        }
        self._state.raw_state["episode_stats"] = episode_stats
        self._state.raw_state = self._snapshot_state()

        return obs

    @property
    def state(self) -> HFTState:
        return self._state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_observation(
        self,
        step_profit: float = 0.0,
        step_loss: float = 0.0,
        latency_ms: float = BASE_LATENCY_MS,
        active_nodes: int = TOTAL_NODES,
        security_level: float = 0.0,
        selected_indices: Optional[List[int]] = None,
    ) -> HFTObservation:
        opportunities = [
            HFTOpportunity(
                value=opp["value"],
                latency_sensitivity=opp["latency_sensitivity"],
                signal_strength=float(opp.get("signal_strength", 0.0)),
                anomaly_score=float(opp.get("anomaly_score", 0.0)),
            )
            for opp in self._current_opportunities
        ]

        total_steps = int(self._state.total_steps)
        current_step = int(self._state.step_count)
        working_hours = 12.0
        hours_per_step = (working_hours / float(total_steps)) if total_steps > 0 else working_hours
        elapsed_hours = min(working_hours, float(current_step) * hours_per_step)
        hours_remaining = max(0.0, working_hours - elapsed_hours)
        session_start = datetime(2000, 1, 1, 9, 0, 0)
        session_time = session_start + timedelta(hours=elapsed_hours)
        time_of_day = session_time.strftime("%H:%M")
        net_profit_for_rate = float(self._state.cumulative_profit - self._state.cumulative_loss)
        target_gap_remaining_for_rate = max(0.0, float(self._task.target_profit) - net_profit_for_rate)
        required_avg_net_per_hour = (
            (target_gap_remaining_for_rate / hours_remaining)
            if hours_remaining > 0.0
            else target_gap_remaining_for_rate
        )

        prompt = self._build_prompt(
            latency_ms=latency_ms,
            active_nodes=active_nodes,
            security_level=security_level,
            selected_indices=list(selected_indices or []),
            step_profit=step_profit,
            step_loss=step_loss,
            opportunities=opportunities,
        )

        return HFTObservation(
            prompt=prompt,
            step=self._state.step_count,
            latency_ms=latency_ms,
            active_nodes=active_nodes,
            security_level=security_level,
            available_opportunities=AVAILABLE_OPPORTUNITIES,
            selected_indices=list(selected_indices or []),
            cumulative_profit=self._state.cumulative_profit,
            cumulative_loss=self._state.cumulative_loss,
            target_profit=self._task.target_profit,
            node_health=list(self._state.node_health),
            node_compromised=list(self._state.node_compromised),
            system_stress=float(self._state.system_stress),
            time_of_day=time_of_day,
            required_avg_net_per_hour=float(required_avg_net_per_hour),
            time_step=int(self._state.step_count),
            max_steps=int(self._state.total_steps),
            opportunities=opportunities,
            step_profit=step_profit,
            step_loss=step_loss,
            info={
                "task_id": self._task.task_id,
                "task_description": self._task.description,
                "graded_tasks": list_graded_tasks(),
            },
            reward=0.0,
            done=False,
            metadata={
                "task_id": self._task.task_id,
                "step": self._state.step_count,
                "total_steps": int(self._state.total_steps),
            },
        )

    def _build_prompt(
        self,
        latency_ms: float,
        active_nodes: int,
        security_level: float,
        selected_indices: List[int],
        step_profit: float,
        step_loss: float,
        opportunities: List[HFTOpportunity],
    ) -> str:
        total_steps = int(self._state.total_steps)
        current_step = int(self._state.step_count)
        net_profit = float(self._state.cumulative_profit - self._state.cumulative_loss)
        target_gap = max(0.0, float(self._task.target_profit) - net_profit)
        working_hours = 12.0
        hours_per_step = (working_hours / float(total_steps)) if total_steps > 0 else working_hours
        elapsed_hours = min(working_hours, float(current_step) * hours_per_step)
        remaining_hours = max(0.0, working_hours - elapsed_hours)
        required_avg_per_hour = (
            (target_gap / remaining_hours) if remaining_hours > 0.0 else target_gap
        )
        session_start = datetime(2000, 1, 1, 9, 0, 0)
        session_time = session_start + timedelta(hours=elapsed_hours)
        session_time_str = session_time.strftime("%H:%M")
        session_end_str = "21:00"

        lines = [
            "=== HFT Security Environment — Intraday Session ===",
            f"Task: {self._task.description}",
            "",
            "System Status:",
            f"  Latency: {latency_ms:.2f} ms",
            f"  Active Nodes: {active_nodes}/{TOTAL_NODES}",
            f"  Security Level: {security_level:.2f}",
            f"  System Stress: {float(self._state.system_stress):.2f}",
            f"  Time Of Day: {session_time_str}",
            f"  Session End Time: {session_end_str}",
            "",
            "Market:",
            f"  Available Opportunities: {AVAILABLE_OPPORTUNITIES}",
            f"  Selected indices (last step): {selected_indices}",
            "",
            "Financials:",
            f"  Cumulative Profit: {self._state.cumulative_profit:.2f}",
            f"  Cumulative Loss: {self._state.cumulative_loss:.2f}",
            f"  Target Profit: {self._task.target_profit:.2f}",
            f"  Required Avg Net/Hour To Hit Target: {required_avg_per_hour:.2f}",
            f"  Last Step Profit: {step_profit:.2f}",
            f"  Last Step Loss: {step_loss:.2f}",
            "",
            "Nodes:",
            "  (health, compromised)",
        ]
        for i in range(TOTAL_NODES):
            h = float(self._state.node_health[i]) if self._state.node_health else 1.0
            c = bool(self._state.node_compromised[i]) if self._state.node_compromised else False
            lines.append(f"  node[{i}] health={h:.2f} compromised={c}")

        lines.extend(
            [
                "",
                "Available Opportunities:",
            ]
        )
        for i, opp in enumerate(opportunities):
            lines.append(
                f"  [{i}] value={opp.value}, signal_strength={opp.signal_strength:.2f}, "
                f"anomaly_score={opp.anomaly_score:.2f}, latency_sensitivity={opp.latency_sensitivity}"
            )
        lines.extend(
            [
                "",
                "Your action: provide security_level (0.0-1.0), active_nodes (1-10), "
                "selected_indices (list of opportunity indices).",
                "Only indices passing the security anomaly threshold will execute.",
            ]
        )
        return "\n".join(lines)

    def _snapshot_state(self) -> Dict[str, Any]:
        # Persist long-lived dynamics across steps.
        return {
            "step": self._state.step_count,
            "cumulative_profit": self._state.cumulative_profit,
            "cumulative_loss": self._state.cumulative_loss,
            "target_profit": self._task.target_profit,
            "task_id": self._task.task_id,
            "episode_seed": int(self._state.raw_state.get("episode_seed", self._task.base_seed)),
            "latency_attack_steps_left": int(self._state.raw_state.get("latency_attack_steps_left", 0)),
            "latency_attack_spike_left": float(self._state.raw_state.get("latency_attack_spike_left", 0.0)),
            "attack_cooldown_left": int(self._state.raw_state.get("attack_cooldown_left", 0)),
            "compromised_steps_left": list(
                self._state.raw_state.get("compromised_steps_left", [0 for _ in range(TOTAL_NODES)])
            ),
            "failure_bias": float(self._state.raw_state.get("failure_bias", 1.0)),
            "episode_stats": dict(self._state.raw_state.get("episode_stats", {})),
        }

    def _compute_reward_signals(
        self, *, action: HFTAction, observation: HFTObservation
    ) -> Dict[str, float]:
        signals: Dict[str, float] = {}
        for provider in self._reward_providers:
            signals.update(provider.compute(action=action, observation=observation))
        return signals
