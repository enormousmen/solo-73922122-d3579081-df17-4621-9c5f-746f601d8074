"""Atomic checkpoint persistence.

A checkpoint is a single ``checkpoint.pt`` file inside the runs directory.
Saves are atomic on POSIX filesystems:

1. serialize to ``.checkpoint.pt.tmp.<uuid>`` in the same directory,
2. flush + fsync the file,
3. ``os.replace()`` it over ``checkpoint.pt`` (atomic rename),
4. fsync the directory so the rename itself is durable.

If the process is killed at any point, ``checkpoint.pt`` is therefore either
the complete previous checkpoint or the complete new one — never a torn file.
A failed save (e.g. disk full) leaves the previous checkpoint untouched so it
can be used for a retry or for recovery after a restart.

TEST-ONLY HOOK: if a file named ``INJECT_SAVE_FAILURES`` exists in the runs
directory, every save fails *before touching anything* (the old checkpoint
stays intact).  ``scripts/verify_repro.py`` uses this to exercise the
save-failure path; it is never present in normal operation.
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import torch

CHECKPOINT_FILENAME = "checkpoint.pt"
INJECT_FAILURES_FILENAME = "INJECT_SAVE_FAILURES"


class CheckpointError(IOError):
    """Raised when a checkpoint could not be persisted."""


def checkpoint_path(runs_dir) -> Path:
    return Path(runs_dir) / CHECKPOINT_FILENAME


def save_checkpoint(runs_dir, payload: dict, save_delay_seconds: float = 0.0) -> Path:
    runs = Path(runs_dir)
    runs.mkdir(parents=True, exist_ok=True)

    # --- test-only fault injection (see module docstring) ---
    if (runs / INJECT_FAILURES_FILENAME).exists():
        raise CheckpointError(
            f"checkpoint save failed: injected fault "
            f"('{INJECT_FAILURES_FILENAME}' file present in {runs})"
        )
    # ---------------------------------------------------------

    final = runs / CHECKPOINT_FILENAME
    tmp = runs / f".{CHECKPOINT_FILENAME}.tmp.{uuid.uuid4().hex}"
    try:
        with open(tmp, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        if save_delay_seconds > 0:
            # Simulates slow storage.  The window between "new file fully
            # written" and "atomic rename" is where a kill must still leave
            # the OLD checkpoint intact.
            time.sleep(save_delay_seconds)
        os.replace(tmp, final)
        _fsync_dir(runs)
    except CheckpointError:
        raise
    except Exception as exc:  # noqa: BLE001 - report any persistence failure
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise CheckpointError(f"failed to persist checkpoint to {final}: {exc!r}") from exc
    return final


def load_checkpoint(runs_dir) -> dict | None:
    """Load the checkpoint if one exists; ``None`` otherwise.

    Only complete checkpoints can be observed thanks to the atomic rename.
    """
    final = checkpoint_path(runs_dir)
    if not final.exists():
        return None
    return torch.load(final, map_location="cpu", weights_only=True)


def cleanup_stale_tmp_files(runs_dir) -> list[str]:
    """Remove leftover temp files from saves interrupted by a kill."""
    removed = []
    runs = Path(runs_dir)
    if runs.is_dir():
        for tmp in runs.glob(f".{CHECKPOINT_FILENAME}.tmp.*"):
            try:
                tmp.unlink()
                removed.append(tmp.name)
            except OSError:
                pass
    return removed


def _fsync_dir(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
