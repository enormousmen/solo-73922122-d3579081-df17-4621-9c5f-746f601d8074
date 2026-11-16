"""FastAPI surface for the single-job resumable training service.

Endpoints (interactive docs at /docs):

    GET  /                     service info
    GET  /config               effective default config + runtime info
    POST /train/start          start a fresh run (optional config overrides, force)
    POST /train/pause          stop training; 200 only after a full checkpoint is persisted
    POST /train/resume         continue from the latest checkpoint / paused state
    GET  /train/status         progress and checkpoint health
    GET  /train/losses         per-batch loss history
    GET  /train/parameters     download model parameters (.pt) from latest checkpoint
    GET  /train/checkpoint     download the full checkpoint (.pt)
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response

from .config import load_default_config
from .trainer import (
    CheckpointSaveError,
    ConflictError,
    JobManager,
    NotFoundError,
    PauseTimeoutError,
)

manager: JobManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    runs_dir = os.environ.get("RUNS_DIR", "runs")
    manager = JobManager(runs_dir=runs_dir, default_config=load_default_config())
    yield


app = FastAPI(
    title="Resumable Training API",
    version="1.0.0",
    description=(
        "Single-job CPU training service (two-layer FC net + Dropout + Adam). "
        "Pause/resume and kill/restart recovery reproduce uninterrupted runs "
        "bit-exactly under the same environment, seed and config."
    ),
    lifespan=lifespan,
)


def _conflict(exc: Exception) -> HTTPException:
    return HTTPException(status_code=409, detail=str(exc))


# TEST-ONLY endpoint (enabled via ENABLE_TEST_CRASH_ENDPOINT=1): hard-exit the
# process with os._exit, simulating SIGKILL / preemption / power loss.  Used by
# scripts/verify_repro.py; never enable in production.
if os.environ.get("ENABLE_TEST_CRASH_ENDPOINT") == "1":
    @app.post("/test/crash", include_in_schema=False)
    def _test_crash() -> None:  # pragma: no cover - kills the process
        os._exit(1)


@app.get("/")
def root() -> dict:
    return {
        "service": "resumable-training-api",
        "docs": "/docs",
        "endpoints": [
            "GET /config",
            "POST /train/start",
            "POST /train/pause",
            "POST /train/resume",
            "GET /train/status",
            "GET /train/losses",
            "GET /train/parameters",
            "GET /train/checkpoint",
        ],
    }


@app.get("/config")
def get_config() -> dict:
    return {
        "default_config": manager.default_config.to_dict(),
        "runs_dir": str(manager.runs_dir),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
    }


@app.post("/train/start")
def start(body: dict[str, Any] | None = Body(default=None)) -> dict:
    """Start a fresh run. Body: {"force": bool, "config": {overrides}}."""
    body = body or {}
    force = bool(body.get("force", False))
    overrides = body.get("config") or {}
    try:
        return manager.start(overrides=overrides, force=force)
    except ConflictError as exc:
        raise _conflict(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/train/pause")
async def pause() -> dict:
    """Stop training; responds only after the trainer stopped AND a complete
    checkpoint is persisted.  Runs in a thread so other queries stay live."""
    try:
        return await asyncio.to_thread(manager.pause)
    except PauseTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except CheckpointSaveError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(exc) from exc


@app.post("/train/resume")
def resume() -> dict:
    """Resume from the paused state / latest checkpoint. Only one trainer can
    run at a time: concurrent resumes yield exactly one 200 and 409s."""
    try:
        return manager.resume()
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConflictError as exc:
        raise _conflict(exc) from exc


@app.get("/train/status")
def status() -> dict:
    return manager.status()


@app.get("/train/losses")
def losses(
    since_step: int | None = Query(default=None, description="only losses with step > this"),
    limit: int | None = Query(default=None, description="max number of entries"),
) -> dict:
    return manager.get_losses(since_step=since_step, limit=limit)


@app.get("/train/parameters")
def parameters() -> Response:
    """Download current model parameters (.pt) from the latest checkpoint."""
    try:
        data = manager.read_parameters()
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="model_parameters.pt"'},
    )


@app.get("/train/checkpoint")
def checkpoint() -> FileResponse:
    """Download the full checkpoint (model + optimizer + RNG + history)."""
    try:
        path = manager.checkpoint_file()
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, filename="checkpoint.pt",
                        media_type="application/octet-stream")
