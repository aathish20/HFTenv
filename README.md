---
title: HFTEnv
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
---

# HFT Security Environment (OpenEnv)

This environment simulates a high-frequency trading operations desk under adversarial cyber conditions.  
The agent must make step-wise control decisions to maximize net profit while maintaining system safety and throughput.

## Motivation

This is a real-world style operations-control task, in a pressured environment:
- trade execution under latency pressure
- security filtering vs throughput tradeoff
- compromised/failing infrastructure nodes
- dynamic adversarial events (fake signals, latency attacks, node compromise)
- time-constrained target delivery (intraday session)

The objective is to reach target profit by session close without destabilizing the system.

## OpenEnv Compliance

The environment implements the OpenEnv interface with typed models:
- typed `Observation` and `Action` via Pydantic (`models.py`)
- `reset()` returns initial observation
- `step(action)` returns `(observation, reward, done, info)` semantics
- `state` is maintained server-side and exposed via environment implementation
- `openenv.yaml` is present at repo root

## Environment Mechanics

The action each step is:

```json
{
  "security_level": 0.0,
  "active_nodes": 1,
  "selected_indices": [0, 2, 5]
}
```

- `security_level` in `[0,1]`: stronger screening vs lower throughput
- `active_nodes` in `[1,10]`: capacity vs cost/attack surface
- `selected_indices`: requested opportunities to execute

Important behavior:
- agent-selected opportunities are not guaranteed to execute
- higher security tends to block more suspicious opportunities
- execution is further capped by working node capacity
- attacks can alter latency, compromise nodes, inject fake opportunities

Detailed walkthrough is documented in:
- [`ENV_WORKING_EXPLAINED.md`](./ENV_WORKING_EXPLAINED.md)

## Observation Space

Each step observation includes:
- `latency_ms`, `system_stress`
- `active_nodes`, `security_level`
- `cumulative_profit`, `cumulative_loss`, `target_profit`
- `time_of_day`, `required_avg_net_per_hour`, `max_steps`
- `node_health[]`, `node_compromised[]`
- `opportunities[]` with:
  - `value`
  - `signal_strength`
  - `anomaly_score`
  - `latency_sensitivity`

## Tasks and Difficulty

Defined in [`tasks.py`](./tasks.py):
- `easy`: stable market, low attack pressure
- `medium`: higher volatility and attack pressure
- `hard`: adversarial market with traps
- `very_hard`: stealth-trap regime with strongest pressure

Current targets:
- easy: `15000`
- medium: `36000`
- hard: `50000`
- very_hard: `50000`

Why `hard` and `very_hard` both use `50000`:
- `very_hard` is made harder primarily through environment dynamics (stronger attack pressure, stealth traps, higher instability risk), not by only increasing the numeric target.
- Keeping the same target isolates difficulty to decision quality under harsher conditions, making task comparison fairer and easier to interpret during evaluation.
- In internal smoke tests, `very_hard` still scores significantly lower than `hard`, confirming it remains meaningfully more difficult even with the same target.

## Deterministic Graders

Task graders are explicit and deterministic in [`graders.py`](./graders.py):
- `grade_easy`
- `grade_medium`
- `grade_hard`
- `grade_very_hard`

Each episode returns:
- `final_score`
- `task_grade` (component + weight breakdown)
- `profit_progress_score`

Score validity:
- final task score is clamped to strict evaluator-safe bounds `[0,1]`

Grader components:
- `profit_progress`
- `loss_control`
- `stability`
- `resilience`
- `ops_continuity`

Harder tasks are profit-dominant and gated by profit progression, so high safety alone cannot yield high final score.

## Reward Function

Per-step reward is meaningful and dense:
- rewards realized profit/progress
- penalizes losses, instability, failed execution, malicious impact, excessive overhead, and zero-execution behavior
- includes terminal shaping at episode end

Auxiliary reward signals are exposed in `info.reward_signals`:
- `hft.profit_ratio`
- `hft.risk_management`
- `hft.cost_efficiency`
- `hft.loss_control`

## Baseline Inference

Submission inference script:
- [`inference.py`](./inference.py)

It uses OpenAI client with required env vars:
- `API_BASE_URL` (default provided)
- `MODEL_NAME` (default provided)
- `HF_TOKEN` (required)

Structured stdout format:
- `[START]`
- `[STEP]`
- `[END]`

By default it runs all tasks (`easy`, `medium`, `hard`, `very_hard`) unless `HFT_TASK_NAME` overrides.

## Smoke Benchmark (Internal Sanity)

A deterministic heuristic baseline was used to sanity-check solvability and task separation.  
Representative multi-seed average (5 seeds):
- easy: score ~`0.998`
- medium: score ~`0.864`
- hard: score ~`0.774`
- very_hard: score ~`0.368`

This confirms:
- easy/medium are solvable
- hard is challenging but reachable
- very_hard remains distinctly difficult

## Run Locally

Start server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Run submission inference:

```bash
python inference.py
```

Optional local test runner (non-submission helper):

```bash
python inference_test.py --task all --runs 1
```

## Docker

Build:

```bash
docker build . -t hftenv
```

Run:

```bash
docker run -p 8000:8000 -e HFT_TASK_ID=easy hftenv
```

## Validation

Run:

```bash
openenv validate
```

And optionally use:
- [`validate-submission.sh`](./validate-submission.sh)

