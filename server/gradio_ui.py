"""
Custom Gradio tab for HFT Security Environment — renders an HFT dashboard.

This module is used as gradio_builder when creating the app; the returned Blocks
appear in the "Custom" tab next to the default "Playground" tab.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import gradio as gr
from openenv.core.env_server.types import EnvironmentMetadata


def _hft_dashboard_html() -> str:
    """Static HFT-themed dashboard HTML for the Custom tab."""
    return """
<div class="hft-dashboard" style="
  font-family: 'Courier New', monospace;
  max-width: 600px;
  margin: 0 auto;
  padding: 20px;
  background: #0a0a0a;
  color: #00ff41;
  border: 1px solid #00ff41;
  border-radius: 8px;
">
  <h2 style="text-align: center; color: #00ff41; border-bottom: 1px solid #00ff41; padding-bottom: 10px;">
    ⚡ HFT Security Environment
  </h2>

  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0;">
    <div style="border: 1px solid #333; padding: 12px; border-radius: 4px;">
      <div style="color: #888; font-size: 0.8rem;">LATENCY</div>
      <div style="font-size: 1.5rem; color: #00ff41;">1.00 ms</div>
    </div>
    <div style="border: 1px solid #333; padding: 12px; border-radius: 4px;">
      <div style="color: #888; font-size: 0.8rem;">ACTIVE NODES</div>
      <div style="font-size: 1.5rem; color: #00ff41;">10 / 10</div>
    </div>
    <div style="border: 1px solid #333; padding: 12px; border-radius: 4px;">
      <div style="color: #888; font-size: 0.8rem;">SECURITY LEVEL</div>
      <div style="font-size: 1.5rem; color: #ffaa00;">0.00</div>
    </div>
    <div style="border: 1px solid #333; padding: 12px; border-radius: 4px;">
      <div style="color: #888; font-size: 0.8rem;">CUM. PROFIT</div>
      <div style="font-size: 1.5rem; color: #00ff41;">$0.00</div>
    </div>
  </div>

  <div style="border: 1px solid #333; padding: 12px; border-radius: 4px; margin: 16px 0;">
    <div style="color: #888; font-size: 0.8rem; margin-bottom: 8px;">MARKET OPPORTUNITIES</div>
    <table style="width: 100%; color: #00ff41; font-size: 0.85rem; border-collapse: collapse;">
      <tr style="border-bottom: 1px solid #333;">
        <th style="text-align: left; padding: 4px;">#</th>
        <th style="text-align: left; padding: 4px;">Value</th>
        <th style="text-align: left; padding: 4px;">Risk</th>
        <th style="text-align: left; padding: 4px;">Latency</th>
      </tr>
      <tr><td style="padding: 4px;">0</td><td>$125</td><td style="color: #ff4444;">0.40</td><td>high</td></tr>
      <tr><td style="padding: 4px;">1</td><td>$100</td><td style="color: #ffaa00;">0.20</td><td>medium</td></tr>
      <tr><td style="padding: 4px;">2</td><td>$100</td><td style="color: #00ff41;">0.05</td><td>low</td></tr>
      <tr><td style="padding: 4px;">3</td><td>$75</td><td style="color: #ffaa00;">0.10</td><td>low</td></tr>
      <tr><td style="padding: 4px;">4</td><td>$50</td><td style="color: #00ff41;">0.05</td><td>low</td></tr>
    </table>
    <div style="color: #555; font-size: 0.75rem; margin-top: 8px;">Showing 5 of 10 opportunities</div>
  </div>

  <p style="text-align: center; color: #555; font-size: 0.85rem; margin-top: 16px;">
    Use the <strong style="color: #00ff41;">Playground</strong> tab to Reset and Step with actions.
    <br>Action format: <code style="color: #ffaa00;">{"security_level": 0.5, "active_nodes": 8, "selected_indices": [0, 2, 5]}</code>
  </p>
</div>
"""


def build_hft_gradio_app(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[EnvironmentMetadata],
    is_chat_env: bool,
    title: str,
    quick_start_md: str,
) -> gr.Blocks:
    """Build the Custom tab Blocks for HFT Security Environment."""
    with gr.Blocks(title=f"{title} — HFT Dashboard") as blocks:
        gr.Markdown(value="# HFT Security Environment Dashboard")
        gr.Markdown(
            value=(
                "This tab shows the **HFT trading dashboard**. Use the **Playground** tab to "
                "Reset and Step with actions like "
                '`{"security_level": 0.5, "active_nodes": 8, "selected_indices": [0, 2, 5]}`.'
            )
        )
        gr.HTML(value=_hft_dashboard_html(), show_label=False)
    return blocks
