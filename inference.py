"""Round-1 baseline inference for HFT Security Environment.

This script emits strict structured stdout logs:
- [START]
- [STEP]
- [END]

Required environment variables:
- API_BASE_URL
- MODEL_NAME
- HF_TOKEN
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

try:
    from hftenv.models import HFTAction
    from hftenv.server.environment import HFTSecurityEnvironment
except ImportError:
    from models import HFTAction  # type: ignore[no-redef]
    from server.environment import HFTSecurityEnvironment


load_dotenv(override=False)

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN", "")
TASK_NAME = os.getenv("HFT_TASK_NAME", "all")
BENCHMARK = os.getenv("HFT_BENCHMARK", "hftenv")
DEFAULT_TASKS = ["easy", "medium", "hard", "very_hard"]

JSON_RE = re.compile(r"\{[\s\S]*\}")

SYSTEM_PROMPT = (
    "You are the operations manager of a high-frequency trading desk under active cyber pressure. "
    "Your mission is to reach target_profit by maximizing net profit (cumulative_profit - cumulative_loss) "
    "before session end. "
    "Environment dynamics to reason about: "
    "security_level is inversely related to trade throughput but directly related to risk containment; "
    "active_nodes increase execution capacity but also increase operational cost and attack surface; "
    "selected_indices define market exposure and can improve profit or increase loss if risk is misjudged. "
    "During an episode, events may occur such as latency spikes, fake signals, node failures, and node compromise. "
    "You must adapt strategy over time: protect when risk surges, but avoid over-defensive actions that starve execution. "
    "Use Time Of Day, Session End Time, and Required Avg Net/Hour from the snapshot to calibrate urgency near close. "
    "Take deliberate risk when needed, but keep the system stable enough to keep trading. "
    "Return ONLY valid JSON with keys: security_level (0..1), active_nodes (1..10), "
    "selected_indices (array of ints). No extra text."
)


def _parse_action(text: str) -> HFTAction:
    default = HFTAction(
        security_level=0.6,
        active_nodes=6,
        selected_indices=[0, 1, 2, 3],
    )
    if not text:
        return default
    m = JSON_RE.search(text)
    if not m:
        return default
    try:
        data: dict[str, Any] = json.loads(m.group(0))
        security_level = float(data.get("security_level", default.security_level))
        active_nodes = int(data.get("active_nodes", default.active_nodes))
        selected_indices = list(data.get("selected_indices", default.selected_indices))
        return HFTAction(
            security_level=security_level,
            active_nodes=active_nodes,
            selected_indices=selected_indices,
        )
    except Exception:
        return default


def log_start(*, task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(*, step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    action_compact = " ".join(action.strip().split()) if action else "{}"
    done_val = str(done).lower()
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action_compact} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(*, success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(
        f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def run_task(client: Any, task_id: str) -> float:
    env = HFTSecurityEnvironment(task_id=task_id)
    obs = env.reset()

    rewards: list[float] = []
    last_reward = 0.0
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, int(getattr(obs, "max_steps", 10)) + 1):
            if obs.done:
                break

            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Previous step reward: {last_reward:+.4f}\n"
                            "Environment snapshot:\n"
                            f"{obs.prompt}\n\n"
                            "Output strict JSON only. "
                            "Choose a balanced action that avoids both over-filtering and under-security."
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=180,
            )
            assistant_text = completion.choices[0].message.content or "{}"
            action = _parse_action(assistant_text)

            obs = env.step(action)
            reward = float(obs.reward or 0.0)
            rewards.append(reward)
            last_reward = reward
            steps_taken = step
            error_msg = None
            if isinstance(obs.info, dict):
                err = obs.info.get("last_action_error")
                if err:
                    error_msg = str(err)

            log_step(
                step=step,
                action=assistant_text,
                reward=reward,
                done=bool(obs.done),
                error=error_msg,
            )

            if obs.done:
                break

        score = float((obs.info or {}).get("final_score", 0.0))
        score = max(0.0, min(1.0, score))
        success = score >= 0.6
    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    if not API_KEY:
        raise RuntimeError("Missing HF_TOKEN.")
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    task_name = TASK_NAME.strip().lower()
    if task_name in {"all", ""}:
        for t in DEFAULT_TASKS:
            _ = run_task(client, t)
    else:
        _ = run_task(client, task_name)


if __name__ == "__main__":
    main()
