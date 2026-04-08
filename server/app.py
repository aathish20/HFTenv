"""FastAPI application entrypoint for the HFT Security Environment."""

from __future__ import annotations

import inspect
import logging
import os

from openenv.core.env_server.http_server import create_app

try:
    from models import HFTAction, HFTObservation  # type: ignore[no-redef]
    from .environment import HFTSecurityEnvironment
    from .gradio_ui import build_hft_gradio_app
except ImportError:
    from hftenv.models import HFTAction, HFTObservation  # type: ignore[no-redef]
    from hftenv.server.environment import HFTSecurityEnvironment  # type: ignore[no-redef]
    from hftenv.server.gradio_ui import build_hft_gradio_app  # type: ignore[no-redef]


task_id = os.getenv("HFT_TASK_ID", "easy")


def create_hft_environment():
    """Factory function that creates HFTSecurityEnvironment with config."""
    return HFTSecurityEnvironment(task_id=task_id)


_logger = logging.getLogger(__name__)
_sig = inspect.signature(create_app)

if "gradio_builder" in _sig.parameters:
    app = create_app(
        create_hft_environment,
        HFTAction,
        HFTObservation,
        env_name="hftenv",
        gradio_builder=build_hft_gradio_app,
    )
else:
    _logger.warning(
        "Installed openenv-core does not support gradio_builder; "
        "custom Gradio tab will not be available."
    )
    app = create_app(
        create_hft_environment,
        HFTAction,
        HFTObservation,
        env_name="hftenv",
    )


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution.

        uv run --project . server
        uv run --project . server --port 8001
        python -m hftenv.server.app
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
