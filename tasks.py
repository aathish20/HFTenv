"""
Task definitions for HFT Security Environment.

Each task defines a market scenario with different risk profiles,
target profits, and opportunity generation parameters.

Tasks:
  easy       ? Stable market, mostly low-risk
  medium     ? Volatile market, balanced risk
  hard       ? Adversarial market, high-risk traps
  very_hard  ? Stealth trap, safe early ? deadly late
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

VALUES: List[int] = [100, 200, 300, 400, 500]
RISKS: List[float] = [0.05, 0.1, 0.2, 0.4]
LATENCY_SENSITIVITIES: List[str] = ["low", "medium", "high"]
LATENCY_PENALTY_MAP: Dict[str, float] = {
    "low": 0.05,
    "medium": 0.15,
    "high": 0.25,
}


@dataclass
class TaskConfig:
    """Configuration for a single task / market scenario."""

    task_id: str
    description: str
    target_profit: float
    base_seed: int
    loss_multiplier: float = 1.0
    risk_value_correlation: float = 0.0  # 0=independent, 1=high-value?high-risk
    # Per-step overrides: step -> weights
    value_weights: Dict[int, List[float]] = field(default_factory=dict)
    risk_weights: Dict[int, List[float]] = field(default_factory=dict)
    latency_weights: Dict[int, List[float]] = field(default_factory=dict)
    risk_correlation_overrides: Dict[int, float] = field(default_factory=dict)
    # Defaults used when no per-step override exists
    default_value_weights: List[float] = field(
        default_factory=lambda: [0.25, 0.25, 0.25, 0.25]
    )
    default_risk_weights: List[float] = field(
        default_factory=lambda: [0.25, 0.25, 0.25, 0.25]
    )
    default_latency_weights: List[float] = field(
        default_factory=lambda: [0.34, 0.33, 0.33]
    )

    # New long-running episode length
    max_steps: int = 10

    # Attack intensity (base probability scale)
    attack_scale: float = 1.0

    # Difficulty scaling knobs (used by the environment)
    attack_base_prob: float = 0.10
    fake_signal_rate: float = 0.10
    node_failure_multiplier: float = 1.0
    anomaly_noise_level: float = 0.0

    # Difficulty differentiation knobs (used by the environment)
    slip_multiplier: float = 0.6
    malicious_penalty_multiplier: float = 2.0

    # Attack cadence and persistence knobs
    attack_progress_mid_bonus: float = 0.20
    attack_progress_late_bonus: float = 0.40
    low_security_attack_bonus: float = 0.30
    attack_cooldown_steps: int = 3
    latency_attack_spike_range: List[float] = field(default_factory=lambda: [45.0, 100.0])
    latency_attack_duration_range: List[int] = field(default_factory=lambda: [2, 4])
    compromise_duration_range: List[int] = field(default_factory=lambda: [4, 6])
    fake_signal_inject_range: List[int] = field(default_factory=lambda: [1, 2])

    # Stress / recovery shaping knobs
    stress_growth_multiplier: float = 1.0
    stress_recovery_base: float = 0.05
    stress_recovery_security_scale: float = 0.10
    failure_bias_recovery: float = 0.90

    # Reward shaping and scoring knobs
    reward_target_threshold: float = 0.60
    terminal_bonus_success: float = 5.0
    terminal_bonus_failure: float = -5.0


def generate_opportunities(
    task: TaskConfig, step: int, count: int = 10
) -> List[Dict[str, Any]]:
    """Generate deterministic opportunities for a given step.

    seed = task.base_seed + step  ?  same seed produces same opportunities.
    """
    seed = task.base_seed + step
    rng = random.Random(seed)

    vw = task.value_weights.get(step, task.default_value_weights)
    rw = task.risk_weights.get(step, task.default_risk_weights)
    lw = task.latency_weights.get(step, task.default_latency_weights)

    # Risk-value correlation: on harder tasks, high value ? high risk
    correlation = task.risk_correlation_overrides.get(
        step, task.risk_value_correlation
    )
    value_to_risk: Dict[int, float] = {100: 0.05, 200: 0.1, 300: 0.2, 400: 0.3, 500: 0.4}

    # Malicious opportunity generation (deterministic via seed)
    # Harder tasks contain more malicious opportunities.
    malicious_rate_by_task: Dict[str, float] = {
        "easy": 0.05,
        "medium": 0.10,
        "hard": 0.20,
        "very_hard": 0.25,
    }
    malicious_rate = malicious_rate_by_task.get(task.task_id, 0.10)

    opportunities: List[Dict[str, Any]] = []
    for _ in range(count):
        # Value: random between 100-500 (discrete buckets for determinism)
        value = rng.choices(VALUES, weights=vw, k=1)[0]
        if correlation > 0.0 and rng.random() < correlation:
            risk = value_to_risk[value]
        else:
            risk = rng.choices(RISKS, weights=rw, k=1)[0]
        lat_sens = rng.choices(LATENCY_SENSITIVITIES, weights=lw, k=1)[0]
        is_malicious = rng.random() < malicious_rate

        # Observable adversarial signals (NOT perfectly separating malicious vs legit)
        # - malicious tends to have higher anomaly_score, but overlaps with legit
        # - signal_strength is a noisy indicator correlated with anomaly
        if is_malicious:
            anomaly_score = min(1.0, max(0.0, rng.uniform(0.55, 0.95)))
        else:
            anomaly_score = min(1.0, max(0.0, rng.uniform(0.05, 0.75)))
        # Task-specific noise: makes anomaly less reliable on harder tasks
        if task.anomaly_noise_level > 0.0:
            anomaly_score = min(
                1.0,
                max(
                    0.0,
                    anomaly_score + rng.uniform(-task.anomaly_noise_level, task.anomaly_noise_level),
                ),
            )

        # signal_strength loosely correlates with anomaly_score, with noise
        signal_strength = min(1.0, max(0.0, 1.0 - anomaly_score + rng.uniform(-0.15, 0.15)))
        # latency_sensitivity: random between 0-1 in spec; we keep categorical for penalty map
        # and also provide a numeric sensitivity in [0,1] for realism.
        latency_sensitivity_num = float(rng.random())

        opportunities.append(
            {
                "value": int(value),
                "risk": float(risk),
                "latency_sensitivity": lat_sens,
                "latency_sensitivity_num": latency_sensitivity_num,
                "is_malicious": bool(is_malicious),
                "signal_strength": float(signal_strength),
                "anomaly_score": float(anomaly_score),
            }
        )
    return opportunities


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASK_EASY = TaskConfig(
    task_id="easy",
    description=(
        "Stable Market: Mostly low-risk opportunities with low latency "
        "sensitivity. Balanced value distribution. Goal: avoid unnecessary cost."
    ),
    # Target tuned to require near-full episode to reach.
    target_profit=15000.0,
    base_seed=42,
    loss_multiplier=1.0,
    risk_value_correlation=0.0,
    default_value_weights=[0.15, 0.20, 0.25, 0.25, 0.15],
    default_risk_weights=[0.50, 0.30, 0.15, 0.05],
    default_latency_weights=[0.60, 0.30, 0.10],
    max_steps=30,
    attack_scale=0.6,
    attack_base_prob=0.10,
    fake_signal_rate=0.05,
    node_failure_multiplier=0.8,
    anomaly_noise_level=0.02,
    slip_multiplier=0.3,
    malicious_penalty_multiplier=1.2,
    attack_progress_mid_bonus=0.10,
    attack_progress_late_bonus=0.20,
    low_security_attack_bonus=0.20,
    attack_cooldown_steps=5,
    latency_attack_spike_range=[35.0, 80.0],
    latency_attack_duration_range=[2, 3],
    compromise_duration_range=[3, 4],
    fake_signal_inject_range=[1, 1],
    stress_growth_multiplier=0.75,
    stress_recovery_base=0.07,
    stress_recovery_security_scale=0.12,
    failure_bias_recovery=0.85,
    reward_target_threshold=0.55,
    terminal_bonus_success=6.0,
    terminal_bonus_failure=-4.0,
)

TASK_MEDIUM = TaskConfig(
    task_id="medium",
    description=(
        "Volatile Market: Mix of medium/high risk with more latency-sensitive "
        "trades. Goal: balance speed vs safety."
    ),
    # Target tuned to require near-full episode to reach.
    target_profit=36000.0,
    base_seed=137,
    loss_multiplier=1.5,
    risk_value_correlation=0.3,
    default_value_weights=[0.10, 0.20, 0.25, 0.25, 0.20],
    default_risk_weights=[0.15, 0.25, 0.35, 0.25],
    default_latency_weights=[0.20, 0.40, 0.40],
    max_steps=50,
    attack_scale=0.9,
    attack_base_prob=0.20,
    fake_signal_rate=0.12,
    node_failure_multiplier=1.0,
    anomaly_noise_level=0.08,
    slip_multiplier=0.6,
    malicious_penalty_multiplier=2.0,
    attack_progress_mid_bonus=0.20,
    attack_progress_late_bonus=0.35,
    low_security_attack_bonus=0.30,
    attack_cooldown_steps=4,
    latency_attack_spike_range=[50.0, 110.0],
    latency_attack_duration_range=[2, 4],
    compromise_duration_range=[4, 6],
    fake_signal_inject_range=[1, 2],
    stress_growth_multiplier=1.0,
    stress_recovery_base=0.05,
    stress_recovery_security_scale=0.10,
    failure_bias_recovery=0.90,
    reward_target_threshold=0.60,
    terminal_bonus_success=5.0,
    terminal_bonus_failure=-5.0,
)

TASK_HARD = TaskConfig(
    task_id="hard",
    description=(
        "Adversarial Market: High-risk opportunities are frequent with high "
        "latency sensitivity and high-value traps. High-value opportunities "
        "tend to carry proportionally higher risk. Goal: avoid loss spikes "
        "while maintaining profit."
    ),
    # Target tuned to require near-full episode to reach.
    target_profit=50000.0,
    base_seed=256,
    loss_multiplier=2.0,
    risk_value_correlation=0.8,
    default_value_weights=[0.05, 0.10, 0.20, 0.30, 0.35],
    default_risk_weights=[0.05, 0.10, 0.35, 0.50],
    default_latency_weights=[0.10, 0.30, 0.60],
    max_steps=70,
    attack_scale=1.2,
    attack_base_prob=0.30,
    fake_signal_rate=0.20,
    node_failure_multiplier=1.2,
    anomaly_noise_level=0.15,
    slip_multiplier=0.9,
    malicious_penalty_multiplier=2.5,
    attack_progress_mid_bonus=0.25,
    attack_progress_late_bonus=0.45,
    low_security_attack_bonus=0.35,
    attack_cooldown_steps=3,
    latency_attack_spike_range=[60.0, 130.0],
    latency_attack_duration_range=[3, 4],
    compromise_duration_range=[5, 7],
    fake_signal_inject_range=[1, 3],
    stress_growth_multiplier=1.2,
    stress_recovery_base=0.04,
    stress_recovery_security_scale=0.09,
    failure_bias_recovery=0.92,
    reward_target_threshold=0.62,
    terminal_bonus_success=5.0,
    terminal_bonus_failure=-6.0,
)

TASK_VERY_HARD = TaskConfig(
    task_id="very_hard",
    description=(
        "Stealth Trap Market: Early steps appear safe with low-risk balanced "
        "opportunities. Later steps contain high-risk high-value traps designed "
        "to punish greedy selection. Goal: avoid greedy selection."
    ),
    # Target tuned to require near-full episode to reach.
    target_profit=50000.0,
    base_seed=512,
    loss_multiplier=2.5,
    risk_value_correlation=0.0,
    # Steps 0-4 use these safe defaults
    default_value_weights=[0.15, 0.20, 0.25, 0.25, 0.15],
    default_risk_weights=[0.45, 0.30, 0.15, 0.10],
    default_latency_weights=[0.50, 0.35, 0.15],
    max_steps=100,
    attack_scale=1.5,
    attack_base_prob=0.40,
    fake_signal_rate=0.40,
    node_failure_multiplier=1.4,
    anomaly_noise_level=0.25,
    slip_multiplier=1.2,
    malicious_penalty_multiplier=3.0,
    attack_progress_mid_bonus=0.30,
    attack_progress_late_bonus=0.50,
    low_security_attack_bonus=0.40,
    attack_cooldown_steps=2,
    latency_attack_spike_range=[70.0, 140.0],
    latency_attack_duration_range=[3, 5],
    compromise_duration_range=[6, 8],
    fake_signal_inject_range=[2, 4],
    stress_growth_multiplier=1.35,
    stress_recovery_base=0.03,
    stress_recovery_security_scale=0.08,
    failure_bias_recovery=0.94,
    reward_target_threshold=0.65,
    terminal_bonus_success=5.0,
    terminal_bonus_failure=-7.0,
)

# Override steps 5-9 for very_hard: suddenly dangerous
for _step in range(5, 10):
    TASK_VERY_HARD.value_weights[_step] = [0.03, 0.07, 0.15, 0.30, 0.45]
    TASK_VERY_HARD.risk_weights[_step] = [0.05, 0.05, 0.30, 0.60]
    TASK_VERY_HARD.latency_weights[_step] = [0.05, 0.25, 0.70]
    TASK_VERY_HARD.risk_correlation_overrides[_step] = 0.9


TASKS: Dict[str, TaskConfig] = {
    "easy": TASK_EASY,
    "medium": TASK_MEDIUM,
    "hard": TASK_HARD,
    "very_hard": TASK_VERY_HARD,
}


def get_task(task_id: str) -> TaskConfig:
    """Return the TaskConfig for the given task_id."""
    if task_id not in TASKS:
        raise ValueError(
            f"Unknown task_id: {task_id!r}. Available: {list(TASKS.keys())}"
        )
    return TASKS[task_id]
