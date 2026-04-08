"""Custom Gradio tab for HFT environment (human-friendly controls)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import gradio as gr
from openenv.core.env_server.types import EnvironmentMetadata


def _extract_observation(data: Dict[str, Any]) -> Dict[str, Any]:
    obs = data.get("observation", data)
    if not isinstance(obs, dict):
        return {}
    return obs


def _action_preview(security_level: float, active_nodes: int, selected_indices: List[str]) -> str:
    idx: List[int] = []
    for x in selected_indices or []:
        try:
            idx.append(int(x))
        except Exception:
            continue
    return json.dumps(
        {
            "security_level": round(float(security_level), 2),
            "active_nodes": int(active_nodes),
            "selected_indices": idx,
        },
        indent=2,
    )


def _format_overview_markdown(data: Dict[str, Any]) -> str:
    obs = _extract_observation(data)
    info = data.get("info", obs.get("info", {}))
    if not isinstance(info, dict):
        info = {}

    lines = ["# HFT Session"]
    if "task_id" in info:
        lines.append(f"- Task: `{info.get('task_id')}`")
    if "reward" in data:
        lines.append(f"- Reward: `{data.get('reward')}`")
    if "done" in data:
        lines.append(f"- Done: `{data.get('done')}`")

    for key in [
        "time_of_day",
        "latency_ms",
        "active_nodes",
        "security_level",
        "system_stress",
        "cumulative_profit",
        "cumulative_loss",
        "target_profit",
        "required_avg_net_per_hour",
    ]:
        if key in obs:
            lines.append(f"- {key}: `{obs.get(key)}`")

    opportunities = obs.get("opportunities", [])
    if isinstance(opportunities, list) and opportunities:
        lines.append("\n## Opportunities (index / value / signal / anomaly / latency)")
        lines.append("| idx | value | signal | anomaly | latency |")
        lines.append("|---:|---:|---:|---:|---|")
        for i, o in enumerate(opportunities):
            if not isinstance(o, dict):
                continue
            lines.append(
                f"| {i} | {o.get('value', '')} | {o.get('signal_strength', '')} | "
                f"{o.get('anomaly_score', '')} | {o.get('latency_sensitivity', '')} |"
            )

    return "\n".join(lines)


def build_hft_gradio_app(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[EnvironmentMetadata],
    is_chat_env: bool,
    title: str,
    quick_start_md: str,
) -> gr.Blocks:
    """Build a practical custom tab with direct reset/step/state controls."""
    _ = action_fields, metadata, is_chat_env, quick_start_md

    async def reset_hft(task_id: str):
        req: Dict[str, Any] = {}
        if task_id:
            req["task_id"] = task_id
        try:
            data = await web_manager.reset_environment(req)
            obs = _extract_observation(data)
            opps = obs.get("opportunities", [])
            n = len(opps) if isinstance(opps, list) else 10
            choices = [str(i) for i in range(n)]
            return (
                _format_overview_markdown(data),
                json.dumps(data, indent=2, sort_keys=True),
                json.dumps(web_manager.get_state(), indent=2, sort_keys=True),
                gr.update(choices=choices, value=[]),
                "Reset complete.",
            )
        except Exception as exc:
            return ("", "", "", gr.update(), f"Error: {exc}")

    async def step_hft(security_level: float, active_nodes: int, selected_indices: List[str]):
        try:
            idx = []
            for x in selected_indices or []:
                try:
                    idx.append(int(x))
                except Exception:
                    continue
            action = {
                "security_level": float(security_level),
                "active_nodes": int(active_nodes),
                "selected_indices": idx,
            }
            data = await web_manager.step_environment(action)
            obs = _extract_observation(data)
            opps = obs.get("opportunities", [])
            n = len(opps) if isinstance(opps, list) else 10
            choices = [str(i) for i in range(n)]
            return (
                _format_overview_markdown(data),
                json.dumps(data, indent=2, sort_keys=True),
                json.dumps(web_manager.get_state(), indent=2, sort_keys=True),
                gr.update(choices=choices),
                "Step executed.",
            )
        except Exception as exc:
            return ("", "", "", gr.update(), f"Error: {exc}")

    def get_state_sync():
        try:
            return json.dumps(web_manager.get_state(), indent=2, sort_keys=True)
        except Exception as exc:
            return f"Error: {exc}"

    with gr.Blocks(title=f"{title} - HFT Helper") as blocks:
        gr.Markdown("# HFT Human Console")
        gr.Markdown(
            "Use **Live Play** for manual interaction. "
            "Use **API Instructions** for Python/curl usage."
        )
        with gr.Tabs():
            with gr.TabItem("Live Play"):
                gr.Markdown(
                    "Use this tab to play manually: Reset -> choose controls -> Step.\n"
                    "Selected indices are clickable checkboxes."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        task_id = gr.Dropdown(
                            choices=["easy", "medium", "hard", "very_hard"],
                            value="easy",
                            label="Task for Reset",
                        )
                        security_level = gr.Slider(0.0, 1.0, value=0.65, step=0.01, label="Security Level")
                        active_nodes = gr.Slider(1, 10, value=6, step=1, label="Active Nodes")
                        selected_indices = gr.CheckboxGroup(
                            choices=[str(i) for i in range(10)],
                            label="Selected Indices",
                            value=[],
                        )
                        action_preview = gr.Code(
                            label="Action Preview",
                            language="json",
                            interactive=False,
                            value='{\n  "security_level": 0.65,\n  "active_nodes": 6,\n  "selected_indices": []\n}',
                        )
                        with gr.Row():
                            reset_btn = gr.Button("Reset")
                            step_btn = gr.Button("Step", variant="primary")
                            state_btn = gr.Button("Get state")
                        status = gr.Textbox(label="Status", interactive=False)
                    with gr.Column(scale=3):
                        session_md = gr.Markdown("# HFT Session\n\nClick Reset to begin.")
                        raw_json = gr.Code(label="Raw JSON response", language="json", interactive=False)
                        state_json = gr.Code(label="State", language="json", interactive=False)

                reset_btn.click(
                    fn=reset_hft,
                    inputs=[task_id],
                    outputs=[session_md, raw_json, state_json, selected_indices, status],
                )
                step_btn.click(
                    fn=step_hft,
                    inputs=[security_level, active_nodes, selected_indices],
                    outputs=[session_md, raw_json, state_json, selected_indices, status],
                )
                state_btn.click(fn=get_state_sync, outputs=[state_json])
                security_level.change(
                    fn=_action_preview,
                    inputs=[security_level, active_nodes, selected_indices],
                    outputs=[action_preview],
                )
                active_nodes.change(
                    fn=_action_preview,
                    inputs=[security_level, active_nodes, selected_indices],
                    outputs=[action_preview],
                )
                selected_indices.change(
                    fn=_action_preview,
                    inputs=[security_level, active_nodes, selected_indices],
                    outputs=[action_preview],
                )

            with gr.TabItem("API Instructions"):
                gr.Markdown(
                    "## API Usage\n"
                    "### Python\n"
                    "```python\n"
                    "from hftenv import HFTAction, HFTEnv\n"
                    "\n"
                    "env = HFTEnv(base_url='http://localhost:8000')\n"
                    "obs = await env.reset()\n"
                    "res = await env.step(HFTAction(security_level=0.65, active_nodes=6, selected_indices=[1,5,6]))\n"
                    "```\n\n"
                    "### cURL\n"
                    "```bash\n"
                    "curl -X POST http://localhost:8000/reset -H \"Content-Type: application/json\" -d \"{}\"\n"
                    "curl -X POST http://localhost:8000/step  -H \"Content-Type: application/json\" -d '{\"action\":{\"security_level\":0.65,\"active_nodes\":6,\"selected_indices\":[1,5,6]}}'\n"
                    "curl http://localhost:8000/state\n"
                    "```\n\n"
                    "### Decision hints\n"
                    "- Higher security can block more selected opportunities.\n"
                    "- More nodes increase throughput but add cost and attack exposure.\n"
                    "- Use time-of-day and required net/hour to adjust late-session aggression."
                )

    return blocks
