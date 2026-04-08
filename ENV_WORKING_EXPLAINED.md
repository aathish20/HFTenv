# HFT Environment Working Explained

This document explains how the current HFT environment works, step by step, with examples from real logs.

## 1) What The Agent Sends

At each step, the agent sends:

```json
{
  "security_level": 0.88,
  "active_nodes": 3,
  "selected_indices": [6, 7, 1, 0]
}
```

Meaning:
- `security_level`: how strict anomaly filtering should be.
- `active_nodes`: requested execution capacity (and cost/attack-surface tradeoff).
- `selected_indices`: which opportunities to attempt.

## 2) What The Environment Shows Before Decision

The observation prompt includes:
- system status: latency, stress, security, active nodes, time of day, session end time
- market opportunities with `value`, `signal_strength`, `anomaly_score`, `latency_sensitivity`
- financial state: cumulative profit/loss, target, required avg net/hour
- node table: each node's health and compromise flag

So the agent is expected to balance:
- profit progress
- risk/security
- node reliability
- time urgency

## 3) Security Filter Logic (Core Rule)

The environment computes:

```text
anomaly_threshold = 0.90 - 0.45 * security_level
```

So:
- `security_level = 0.00` => threshold `0.90` (very permissive)
- `security_level = 1.00` => threshold `0.45` (strict)

A selected opportunity is allowed only if:

```text
anomaly_score <= anomaly_threshold
```

Important:
- opportunities are still visible in observation
- security decides what is *allowed to execute*, not what is visible

## 4) Example Using Your Hard Log Step

From your log step:
- action: `security_level=0.88`, `selected_indices=[6,7,1,0]`
- threshold = `0.90 - 0.45*0.88 = 0.504`

Anomalies:
- idx 6 -> `0.28` allowed
- idx 7 -> `0.17` allowed
- idx 1 -> `0.10` allowed
- idx 0 -> `0.57` blocked

So final selected after security filter:
- `[6,7,1]`

That matches your prompt line:
- `Selected indices (last step): [6, 7, 1]`

## 5) Additional Filtering: Slip Logic

Even after passing security filter, suspicious trades can be blocked probabilistically at low security:
- applies mainly when `anomaly_score > 0.6`
- lower security => higher slip/block chance

In your example, selected anomalies were low (`<= 0.28`), so slip had little/no effect.

## 6) Capacity Limit (Execution Constraint)

Executed trades are capped by working nodes:

```text
executed_trades = min(len(selected_after_filter), working_nodes)
```

So if:
- selected after filter = 5
- working nodes = 3
- only first 3 execute, others are missed

## 7) Attacks And Dynamics Over Time

The environment injects adversarial effects:
- latency attacks (spikes, multi-step)
- node compromise events
- fake signal injections

Attack pressure increases later in the episode and at low security.

This is why time-of-day and required net/hour matter: near session end, the agent may need controlled aggression.

## 8) Reward Shape (Per Step)

Step reward is a weighted mix of:
- positive: realized profit and progress
- negative: losses, missed trades, idle/over-defense penalties, instability-related penalties

The environment also exposes `reward_signals` in `info`, such as:
- `hft.profit_ratio`
- `hft.risk_management`
- `hft.cost_efficiency`
- `hft.loss_control`

## 9) Final Score (Episode Grade)

Final score is computed by task-specific grader logic in `graders.py`.

Current behavior:
- score is in strict range `[0.002, 0.998]`
- hard/very_hard are strongly tied to profit progression (with gating), so being far from target keeps score low

## 10) Practical Strategy For Agent

A good policy generally does this:
- early: exploit low anomaly + high signal opportunities, keep moderate security
- mid: react to attacks/nodes, avoid both over-filtering and reckless selection
- late: use time-of-day urgency and required net/hour to increase calibrated risk
- always: keep enough active nodes for throughput without opening too much attack surface

## 11) Debug Checklist (When Behavior Looks Wrong)

If selected indices disappear:
- check threshold from security level
- compare each selected index anomaly vs threshold

If reward drops suddenly:
- check latency/stress spikes
- check node failures/compromises
- check over-strict security causing zero-execution steps

If final score feels too high/low:
- inspect `task_grade.components` and `weights` in final `info`
- compare `profit_progress` with target gap and episode end time

