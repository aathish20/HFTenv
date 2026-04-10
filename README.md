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

HFT Security Environment is a multi-step operations-control benchmark for agents acting in a latency-sensitive trading infrastructure under cyber and reliability pressure.

The agent does **not** simply “pick profitable trades.” It must continuously balance:

- execution throughput vs security strictness
- node capacity vs operational cost and failure exposure
- short-term profit vs long-term system stability
- aggressive action vs resilience under adversarial conditions
- progress toward an intraday target vs avoiding infrastructure collapse

This makes the environment a benchmark for **real-world control under uncertainty**, not a toy market simulator.

---

## Why this environment exists

Many agent benchmarks test one-shot reasoning, retrieval, or simple workflow automation. Real operational systems are different:

- actions have delayed consequences
- unsafe optimization can degrade future performance
- infrastructure can become partially compromised
- defensive controls reduce risk but also reduce throughput
- the correct decision depends on current system state, not only immediate reward

This environment is built to capture that class of problem.

The domain is high-frequency trading operations because it naturally combines:

- strict latency sensitivity
- security filtering under adversarial pressure
- dynamic infrastructure health
- noisy or deceptive opportunities
- fixed-session performance targets

The result is a sequential decision-making benchmark where the agent must manage both **economic performance** and **system integrity** over time.

---

## Real-world dynamics being simulated

Although compact, the environment models several dynamics that appear in real production systems:

### 1. Security–throughput tradeoff
Increasing security improves screening and reduces the chance that suspicious opportunities slip through, but it also increases operational overhead and latency. In real systems, stricter controls often improve safety while reducing throughput or increasing false positives.

### 2. Capacity–risk tradeoff
Activating more nodes increases execution capacity, but it also raises cost, expands attack surface, and creates more opportunities for failure. More resources are not automatically better if the system is already under stress.

### 3. Persistent infrastructure degradation
Nodes do not reset to perfect condition every step. Health degrades over time, compromised nodes remain risky for multiple steps, and failures reduce effective capacity. This forces the agent to think beyond immediate reward.

### 4. Adversarial and deceptive events
The environment injects latency attacks, fake signals, and node compromises. These events are not purely cosmetic: they affect latency, execution quality, selection outcomes, and cumulative score. The agent must operate under uncertainty rather than assuming all observations are clean.

### 5. Time-constrained operational pressure
Episodes represent an intraday session with a fixed target profit and a limited number of steps. The observation includes time-of-day and required average net profit per hour, so the agent must adapt strategy as the session progresses. Conservative play may be safe but insufficient; reckless play may be profitable briefly but destabilizing overall.

### 6. Non-guaranteed execution
Selecting an opportunity does not guarantee it will execute successfully. Execution is filtered by security policy, constrained by working node capacity, and affected by infrastructure condition. This matches real operational settings where intent and outcome are not identical.

---

## What the agent controls

At each step, the agent outputs:

```json
{
  "security_level": 0.65,
  "active_nodes": 6,
  "selected_indices": [1, 5, 6]
}
```

### Action fields

- `security_level` in `[0,1]`  
  Controls screening strictness. Higher values reduce risk exposure but increase overhead and can block more opportunities.

- `active_nodes` in `[1,10]`  
  Controls capacity. More nodes can increase execution volume, but also raise cost and failure/attack exposure.

- `selected_indices`  
  The opportunities the agent wants to execute this step.

This action space is deliberately compact: the difficulty comes from **decision quality under interacting dynamics**, not from a bloated interface.

---

## What the agent observes

Each observation exposes both financial and operational state, including:

- `latency_ms`
- `system_stress`
- `active_nodes`
- `security_level`
- `cumulative_profit`
- `cumulative_loss`
- `target_profit`
- `time_of_day`
- `required_avg_net_per_hour`
- `max_steps`
- `node_health[]`
- `node_compromised[]`
- `opportunities[]`

Each opportunity includes:

- `value`
- `signal_strength`
- `anomaly_score`
- `latency_sensitivity`

This gives the agent enough signal to reason about both **profitability** and **system condition**, rather than optimizing blindly for immediate value.

---

## Episode mechanics

A typical step involves:

1. New opportunities are generated for the current market tick.
2. Security policy determines which selected opportunities are allowed.
3. Cyber events may inject fake signals, compromise nodes, or create latency spikes.
4. Node failures and health determine how much execution capacity is actually available.
5. Executed opportunities generate profit or loss depending on latency, anomaly exposure, and system quality.
6. Stress, compromise state, node health, and other persistent factors carry forward.

This means the agent is solving a real **closed-loop control problem**:
its actions affect not only current reward, but also the future operating condition of the environment.

---

## Tasks and difficulty

Tasks are defined in `tasks.py` and increase difficulty by changing environment dynamics, not just raising a score target.

- **easy**  
  Stable regime with lower attack pressure and more forgiving dynamics.

- **medium**  
  Higher volatility, more operational friction, and more frequent adverse events.

- **hard**  
  Adversarial regime where poor security and poor capacity decisions are punished more sharply.

- **very_hard**  
  Strongest pressure, stealthier traps, and harsher resilience requirements.

### Current targets

- `easy`: `15000`
- `medium`: `36000`
- `hard`: `50000`
- `very_hard`: `50000`

`hard` and `very_hard` intentionally share the same target profit.  
The difference is not “bigger number = harder task.” The difference is that `very_hard` demands better policy quality under tougher dynamics. This isolates difficulty in decision-making and robustness rather than score inflation.

---

## Why this is a serious test for agents 

This environment is not a game and not a one-step scoring wrapper.

It has:

- persistent hidden risk realized over multiple steps
- infrastructure degradation over time
- action-dependent future reliability
- meaningful tradeoffs between safety and throughput
- adversarial events that alter system behavior
- deterministic task grading plus dense intermediate reward
- full-episode objectives rather than single-step wins

A toy environment can usually be solved by maximizing immediate reward greedily. This environment is specifically designed so that naive greed can backfire by increasing stress, reducing future execution quality, or allowing malicious opportunities through.

---

## Deterministic graders

Task graders are implemented in `graders.py` and are explicit and deterministic. Each completed episode returns:

- `final_score`
- `task_grade`
- `profit_progress_score`

The final score is clamped to evaluator-safe bounds `[0,1]`. Grading components include:

- `profit_progress`
- `loss_control`
- `stability`
- `resilience`
- `ops_continuity`

Harder tasks remain profit-dominant and are gated by profit progression, so an agent cannot score well by being “safe but ineffective.”

---

## Reward design

The environment provides dense per-step reward rather than only terminal success/failure. Reward reflects:

- realized progress toward profitable execution
- operational efficiency
- losses from malicious or poor-quality execution
- failed execution and instability
- excessive security overhead
- zero-execution or under-utilization behavior
- terminal shaping at episode end

Auxiliary reward signals are exposed in `info.reward_signals`, including:

- `hft.profit_ratio`
- `hft.risk_management`
- `hft.cost_efficiency`
- `hft.loss_control`

This makes the environment useful both for evaluation and for agent training.

---

## OpenEnv compliance

The environment implements the OpenEnv interface with typed Pydantic models:

- typed `Observation` and `Action`
- `reset()` returns the initial observation
- `step(action)` returns updated observation, reward, done, and info semantics
- server-side environment state
- `openenv.yaml` at the repository root

The repository also includes Docker packaging and a baseline inference script.

---

## Baseline inference

The submission inference script is:

- `inference.py`

It uses the OpenAI client and reads required configuration from environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Structured stdout logs follow the required format:

- `[START]`
- `[STEP]`
- `[END]`

By default, inference runs across all tasks unless `HFT_TASK_NAME` is provided.

---

## Internal sanity benchmark

A deterministic heuristic baseline was used to sanity-check solvability and task separation.

Representative multi-seed average (5 seeds):

- `easy`: score ~`0.998`
- `medium`: score ~`0.864`
- `hard`: score ~`0.774`
- `very_hard`: score ~`0.368`

This indicates:

- easy and medium are solvable
- hard is challenging but reachable
- very_hard remains distinctly more difficult

---

## Run locally

Start the server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Run submission inference:

```bash
python inference.py
```

Optional local validation helper:

```bash
./validate-submission.sh
```

---

## Docker

Build:

```bash
docker build . -t hftenv
```

Run:

```bash
docker run -p 8000:8000 -e HFT_TASK_ID=easy hftenv
```

---

## Validation

Run:

```bash
openenv validate
```

You can also use:

- `validate-submission.sh`
