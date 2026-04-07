---
title: HFTEnv
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
---

# HFT Security Environment

OpenEnv-compatible environment for high-frequency trading under cyber attack pressure.

## What This Simulates

An agent must optimize profit while balancing:
- Security controls (`security_level`)
- Infrastructure capacity (`active_nodes`)
- Trade selection (`selected_indices`)

The environment includes:
- Dynamic market opportunities
- Adversarial effects (latency attacks, fake signals, node compromise)
- Node health and failure dynamics
- Multi-step stress accumulation and recovery

## Action Space

```json
{
  "security_level": 0.0,
  "active_nodes": 1,
  "selected_indices": [0, 2, 5]
}
```

- `security_level`: float in `[0.0, 1.0]`
- `active_nodes`: int in `[1, 10]`
- `selected_indices`: list of opportunity indices to execute

## Observation Space

Main fields in each step observation:
- `step`, `max_steps`
- `latency_ms`
- `active_nodes`, `security_level`
- `cumulative_profit`, `cumulative_loss`, `target_profit`
- `node_health`, `node_compromised`, `system_stress`
- `opportunities[]`:
  - `value`
  - `signal_strength`
  - `anomaly_score`
  - `latency_sensitivity`

## Tasks

Defined in [`tasks.py`](./tasks.py):
- `easy`
- `medium`
- `hard`
- `very_hard`

Round-1 required minimum is 3 tasks (`easy`, `medium`, `hard`).

## Scoring

At episode end, environment emits:
- `final_score` in `[0.0, 1.0]` from deterministic task grader (`graders.py`)
- `task_grade` with full component and weight breakdown
- `profit_progress_score` as normalized net-profit progress (for diagnostics)
- `raw_final_reward` for debugging/training visibility

`profit_progress_score` formula:
- `net_profit = max(cumulative_profit - cumulative_loss, 0.0)`
- `profit_progress_score = clamp(net_profit / target_profit, 0.0, 1.0)`

Deterministic grader:
- Implemented in [`graders.py`](./graders.py) as `grade_episode(...)`
- Returns score in `[0.0, 1.0]` for each task (`easy`, `medium`, `hard`, `very_hard`)
- Uses transparent weighted components:
  - `profit_progress`
  - `loss_control`
  - `stability`
  - `resilience`
  - `ops_continuity`

## Inference Script

Use [`inference.py`](./inference.py). It emits strict structured logs:
- `[START]`
- `[STEP]`
- `[END]`

Required env vars (submission validator expectations):
- `API_BASE_URL` (LLM API endpoint)
- `MODEL_NAME` (model id)
- `HF_TOKEN` (API key)

## Local Run

Run server:

```bash
cd hftenv
HFT_TASK_ID=easy uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Run inference:

```bash
python inference.py
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
