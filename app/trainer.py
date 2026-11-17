"""Single-job training manager.

One background trainer thread runs the loop below; FastAPI handlers only
manipulate flags / wait on conditions, so slow training or slow checkpoint
saves never block status queries.

Pause/resume/start handoff (the invariants this module enforces):

* While a checkpoint save is in flight (``saving``), ``resume`` and
  ``start`` wait for it to finish (bounded, then 409) — they never mutate
  or replace the training state mid-save, so a late save of an old run can
  never overwrite a new run's checkpoint or health state.
* Concurrent pauses share the outcome of ONE save: one request becomes the
  saver, the rest wait for that save's conclusion and all report the same
  persisted/failed result for THEIR run — never a later run's state.
* A failed pause-save keeps the previous checkpoint intact and may be
  retried by a later pause; a run discarded by ``force`` start is never
  resurrected (its checkpoint file is removed only after no save of it can
  still be in flight).

Reproducibility design (bit-exact resume):

* ``torch.set_num_threads(1)`` and deterministic algorithms are enabled, so a
  given sequence of ops always produces the same float32 bits on this build.
* A fresh run seeds the global torch RNG once (``torch.manual_seed(seed)``).
  Model init and Dropout then draw from that single stream.
* The per-epoch sample order is a pure function of ``(seed, epoch)`` via a
  dedicated ``random.Random`` instance, so the shuffle of any epoch can be
  re-derived after a restart without storing it.
* Every checkpoint stores: model + Adam state, the torch RNG state, the
  position (``epoch``, ``next_batch``), the full loss history, the resolved
  config and a ``completed`` flag.  Checkpoints are only taken at batch
  boundaries, capturing the exact state "ready to run batch k".
* Resume (after pause, or after a process kill + restart) restores all of the
  above and simply continues the same deterministic sequence, so every
  subsequent batch loss and the final parameters are bit-identical to an
  uninterrupted run.
"""
from __future__ import annotations

import io
import random
import threading
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import torch
import torch.nn.functional as F

from .checkpoint import (
    CheckpointError,
    checkpoint_path,
    cleanup_stale_tmp_files,
    load_checkpoint,
    save_checkpoint,
)
from .config import TrainConfig
from .dataset import load_classification_csv
from .model import build_model


class JobState(str, Enum):
    IDLE = "idle"          # no run exists
    RUNNING = "running"    # trainer thread is working through batches
    PAUSING = "pausing"    # pause requested; finishing batch + checkpoint
    PAUSED = "paused"      # trainer stopped; resumable (see checkpoint_dirty)
    FINISHED = "finished"  # all epochs done and final checkpoint persisted
    FAILED = "failed"      # unrecoverable error (see last_error)


class ConflictError(Exception):
    """409 - request incompatible with current state."""


class NotFoundError(Exception):
    """404 - no run / checkpoint exists."""


class PauseTimeoutError(Exception):
    """504 - trainer did not stop within pause_timeout_seconds."""


class CheckpointSaveError(Exception):
    """500 - pause could not persist a complete checkpoint."""


def epoch_permutation(seed: int, epoch: int, n: int) -> list[int]:
    """Deterministic per-epoch shuffle, re-derivable after a restart."""
    rng = random.Random(f"{seed}:{epoch}")
    perm = list(range(n))
    rng.shuffle(perm)
    return perm


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(self, runs_dir: str, default_config: TrainConfig):
        # Bit-exact CPU reproducibility: fixed thread count + deterministic ops.
        torch.set_num_threads(1)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:  # pragma: no cover - depends on torch build
            pass

        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.default_config = default_config

        self._cond = threading.Condition()      # guards all mutable fields below
        self._save_lock = threading.Lock()      # serializes checkpoint writes
        self._pause_requested = False
        self._thread: threading.Thread | None = None
        self._resume_rng_state: torch.Tensor | None = None

        self._reset_runtime()
        self._load_dataset()
        self._recover_from_disk()

    # ------------------------------------------------------------------ setup

    def _reset_runtime(self) -> None:
        self.state = JobState.IDLE
        self.cfg: TrainConfig | None = None
        self.run_id: str | None = None
        self.model = None
        self.optimizer = None
        self.epoch = 0            # epoch of the NEXT batch to run
        self.next_batch = 0       # index (within epoch) of the NEXT batch to run
        self.losses: list[dict] = []
        self.completed = False
        self.recovered = False
        self.saving = False
        self.checkpoint_dirty = False   # in-memory progress newer than disk
        self.last_error: str | None = None
        self.last_checkpoint_at: str | None = None
        self.started_at: str | None = None
        self._saved_point: tuple[int, int] | None = None
        self._resume_rng_state = None

    def _load_dataset(self) -> None:
        path = self.default_config.dataset_path
        self.X, self.y, self.feature_cols = load_classification_csv(path)
        self.num_samples = int(self.X.shape[0])
        self.input_dim = int(self.X.shape[1])
        self.num_classes = int(self.y.max().item()) + 1
        self.dataset_path = path

    def _build_training_objects(self) -> None:
        self.model = build_model(self.input_dim, self.cfg.hidden_dim,
                                 self.cfg.dropout_p, self.num_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    def _recover_from_disk(self) -> None:
        removed = cleanup_stale_tmp_files(self.runs_dir)
        if removed:
            print(f"[trainer] removed stale temp checkpoint files: {removed}")
        ckpt = load_checkpoint(self.runs_dir)
        if ckpt is None:
            return
        self.cfg = TrainConfig.from_dict(ckpt["config"])
        self._build_training_objects()
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        torch.set_rng_state(ckpt["rng_state"])
        self._resume_rng_state = ckpt["rng_state"]
        self.epoch = int(ckpt["epoch"])
        self.next_batch = int(ckpt["next_batch"])
        self.losses = list(ckpt["losses"])
        self.completed = bool(ckpt["completed"])
        self.run_id = ckpt["run_id"]
        self.started_at = ckpt.get("started_at")
        self._saved_point = (self.epoch, self.next_batch)
        try:
            mtime = checkpoint_path(self.runs_dir).stat().st_mtime
            self.last_checkpoint_at = datetime.fromtimestamp(
                mtime, timezone.utc).isoformat()
        except OSError:
            pass
        self.state = JobState.FINISHED if self.completed else JobState.PAUSED
        self.recovered = True
        print(f"[trainer] recovered checkpoint run={self.run_id} "
              f"epoch={self.epoch} next_batch={self.next_batch} "
              f"completed={self.completed}")

    # --------------------------------------------------------------- commands

    def start(self, overrides: dict | None = None, force: bool = False) -> dict:
        with self._cond:
            # Never mutate/replace the training state while a checkpoint save
            # is in flight (e.g. a retried pause-save): wait it out (bounded)
            # so a late save of the old run can never land on the new run's
            # checkpoint.  RUNNING/PAUSING still conflicts immediately.
            self._wait_out_save_and_training()
            if self.run_id is not None and not force:
                raise ConflictError(
                    "a run already exists; use POST /train/resume to continue it "
                    "or POST /train/start with force=true to discard it")
            cfg = self.default_config.merged(overrides or {})
            if cfg.dataset_path != self.default_config.dataset_path:
                raise ValueError("dataset_path is service-level and cannot be overridden")

            self._reset_runtime()
            self.cfg = cfg
            torch.manual_seed(cfg.seed)          # seed model init + dropout stream
            self._build_training_objects()
            self.run_id = uuid.uuid4().hex[:12]
            self.started_at = _now_iso()
            # A fresh run replaces any previous checkpoint on disk.
            checkpoint_path(self.runs_dir).unlink(missing_ok=True)
            self.state = JobState.RUNNING
            self._pause_requested = False
            self._thread = threading.Thread(target=self._train_loop,
                                            name="trainer", daemon=True)
            self._thread.start()
            return self.status()

    def resume(self) -> dict:
        with self._cond:
            # Wait out any in-flight checkpoint save (bounded) instead of
            # mutating the training state mid-save; RUNNING/PAUSING conflicts
            # immediately.
            self._wait_out_save_and_training()
            if self.run_id is None or self.cfg is None:
                raise NotFoundError("no run or checkpoint to resume from")
            if self.state == JobState.FINISHED:
                raise ConflictError(
                    "run already finished; start a new one with force=true")
            # PAUSED or FAILED: continue from the in-memory state, which is the
            # checkpoint state (or newer, if the last save failed).
            if self._resume_rng_state is not None:
                torch.set_rng_state(self._resume_rng_state)
            self.state = JobState.RUNNING
            self._pause_requested = False
            self.last_error = None
            self._thread = threading.Thread(target=self._train_loop,
                                            name="trainer", daemon=True)
            self._thread.start()
            return self.status()

    def _save_wait_timeout(self) -> float:
        """Upper bound (seconds) for start/resume to wait out an in-flight save."""
        if self.cfg is not None:
            return self.cfg.pause_timeout_seconds
        return 30.0

    def _wait_out_save_and_training(self) -> None:
        """Must be called under ``self._cond``.

        Returns only when no trainer is running AND no checkpoint save is in
        flight, so the caller can safely mutate or replace the training state.
        RUNNING/PAUSING conflicts immediately; an in-flight save is waited out
        for a bounded time, then rejected with a conflict (explicit refusal).
        """
        deadline = time.monotonic() + self._save_wait_timeout()
        while True:
            if self.state in (JobState.RUNNING, JobState.PAUSING):
                raise ConflictError("training is already running")
            if not self.saving:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ConflictError(
                    "a checkpoint save is still in progress; "
                    "retry the request in a moment")
            self._cond.wait(remaining)

    def pause(self) -> dict:
        """Stop training and return only after a full checkpoint is persisted.

        Concurrent pauses share the outcome of ONE save: the first becomes the
        saver, the others wait for that save's conclusion — all of them then
        report the same persisted (200) / failed (500) result for THEIR run,
        never a later run's state.  While the save is in flight, resume/start
        wait (or 409), so the stopped training state is never mutated mid-save.
        """
        with self._cond:
            if self.state == JobState.RUNNING:
                self._pause_requested = True
                self.state = JobState.PAUSING
            elif self.state in (JobState.PAUSING, JobState.PAUSED):
                pass  # join an in-flight pause, or retry a failed pause-save
            else:
                raise ConflictError(f"cannot pause while state={self.state.value}")

            timeout = self.cfg.pause_timeout_seconds if self.cfg else 30.0
            deadline = time.monotonic() + timeout
            waited_on_save = False

            while True:
                # Phase 1: wait for the trainer thread to stop.
                while self.state == JobState.PAUSING:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise PauseTimeoutError(
                            "trainer did not stop within "
                            f"{timeout}s; pause is still in progress, "
                            "poll GET /train/status")
                    self._cond.wait(remaining)

                if self.state == JobState.FINISHED:
                    # The last batch completed before the pause landed; the run
                    # is stopped and the final checkpoint is persisted.
                    return self.status()
                if self.state == JobState.RUNNING:
                    # A concurrent resume restarted the trainer before this
                    # pause could save; re-request the pause and wait again.
                    self._pause_requested = True
                    self.state = JobState.PAUSING
                    continue
                # state == PAUSED: trainer stopped.  Ensure the checkpoint is
                # persisted (retrying the save if the trainer's pause-save
                # failed and left the checkpoint dirty).
                if not self.checkpoint_dirty:
                    return self.status()
                if self.saving:
                    # A concurrent pause is already retrying the save: wait for
                    # its outcome instead of starting a competing save.
                    waited_on_save = True
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise PauseTimeoutError(
                            "a checkpoint save is still in progress; "
                            "poll GET /train/status")
                    self._cond.wait(remaining)
                    continue
                if waited_on_save:
                    # The save we waited on concluded and the checkpoint is
                    # still dirty: it failed.  Concurrent pauses share that
                    # conclusion; a later pause request may retry.
                    raise CheckpointSaveError(
                        "training is stopped but the checkpoint could not be "
                        "persisted; the previous checkpoint is intact on disk "
                        "and the pause can be retried. "
                        f"cause: {self.last_error or 'checkpoint save failed'}")
                # Become the saver: mark saving under the lock so resume/start
                # wait, then save outside the lock (queries stay responsive).
                self.saving = True
                break

        # Saver path (outside the lock): persist the stopped trainer's state.
        # keep_saving=True defers the saving=False + notify to this thread so
        # a queued resume/start cannot overtake this pause's response.
        try:
            self._save_checkpoint(keep_saving=True)
        except CheckpointError as exc:
            with self._cond:
                self.saving = False
                self._cond.notify_all()
            raise CheckpointSaveError(
                f"training is stopped but the checkpoint could not be "
                f"persisted; the previous checkpoint is intact on disk and "
                f"the pause can be retried. cause: {exc}") from exc
        with self._cond:
            self.saving = False
            # Capture the conclusion for the run THIS pause stopped.  Even if
            # a queued resume/start flips the state right after, this request
            # still returns its own run's paused state.
            result = self.status()
            self._cond.notify_all()
        return result

    # ------------------------------------------------------------- train loop

    def _train_loop(self) -> None:
        try:
            cfg = self.cfg
            n = self.num_samples
            bs = cfg.batch_size
            batches_per_epoch = (n + bs - 1) // bs

            while self.epoch < cfg.epochs:
                epoch = self.epoch
                perm = epoch_permutation(cfg.seed, epoch, n)
                for b in range(self.next_batch, batches_per_epoch):
                    idx = perm[b * bs:(b + 1) * bs]
                    xb = self.X[idx]
                    yb = self.y[idx]
                    self.model.train()
                    self.optimizer.zero_grad(set_to_none=True)
                    logits = self.model(xb)
                    loss = F.cross_entropy(logits, yb)
                    loss.backward()
                    self.optimizer.step()
                    loss_value = loss.item()
                    if cfg.batch_sleep_seconds > 0:
                        time.sleep(cfg.batch_sleep_seconds)

                    with self._cond:
                        self.next_batch = b + 1
                        self.losses.append({
                            "step": len(self.losses) + 1,
                            "epoch": epoch,
                            "batch": b,
                            "loss": loss_value,
                        })
                        pause_now = self._pause_requested

                    if (b + 1) % cfg.checkpoint_every == 0 or b == batches_per_epoch - 1:
                        try:
                            self._save_checkpoint()
                        except CheckpointError:
                            pass  # recorded in status; retried at next boundary

                    if pause_now:
                        try:
                            self._save_checkpoint()
                        except CheckpointError:
                            pass  # pause() reports the failure to the caller
                        with self._cond:
                            self._resume_rng_state = torch.get_rng_state()
                            self._pause_requested = False
                            self.state = JobState.PAUSED
                            self._cond.notify_all()
                        return

                with self._cond:
                    self.epoch = epoch + 1
                    self.next_batch = 0

            with self._cond:
                self.completed = True
            try:
                self._save_checkpoint(force=True)
            except CheckpointError:
                with self._cond:
                    self.state = JobState.FAILED
                    self._cond.notify_all()
                return
            with self._cond:
                self.state = JobState.FINISHED
                self._cond.notify_all()
        except Exception as exc:  # noqa: BLE001 - never lose the error
            with self._cond:
                self.last_error = f"training failed: {exc!r}"
                self.state = JobState.FAILED
                self._cond.notify_all()

    # -------------------------------------------------------------- checkpoint

    def _save_checkpoint(self, force: bool = False, keep_saving: bool = False) -> None:
        """Persist the current training state.

        The payload is snapshotted under ``self._cond`` and ``saving`` is set
        in the same critical section, so start/resume can never slip in
        between and mutate the state being saved.  The (possibly slow) write
        itself happens outside ``self._cond`` — queries stay responsive — and
        is serialized through ``self._save_lock``.

        If the run was replaced while the save was pending (cannot happen in
        practice because start/resume wait for ``saving``; kept as a safety
        net), the stale save is discarded instead of overwriting the new run's
        checkpoint or health state.

        ``keep_saving=True`` (used by the pause saver) leaves ``saving=True``
        and skips the notify on completion; the caller then captures the
        paused status and releases waiters itself, so a queued resume/start
        cannot overtake the pause's own response.
        """
        with self._cond:
            point = (self.epoch, self.next_batch)
            if not force and not self.checkpoint_dirty and self._saved_point == point:
                return  # this exact point is already persisted
            run_id = self.run_id
            save_delay = self.cfg.checkpoint_sleep_seconds
            payload = {
                "run_id": run_id,
                "config": self.cfg.to_dict(),
                "epoch": self.epoch,
                "next_batch": self.next_batch,
                "completed": self.completed,
                "losses": list(self.losses),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "started_at": self.started_at,
                "saved_at": _now_iso(),
            }
            # Set only after the snapshot succeeded; still the same critical
            # section, so start/resume cannot interleave either way.
            self.saving = True
        try:
            with self._save_lock:
                with self._cond:
                    if self.run_id != run_id:
                        raise CheckpointError(
                            "run was replaced while the save was pending; "
                            "discarding the stale save")
                save_checkpoint(self.runs_dir, payload,
                                save_delay_seconds=save_delay)
        except CheckpointError as exc:
            with self._cond:
                if self.run_id == run_id:
                    self.checkpoint_dirty = True
                    self.last_error = (f"checkpoint save failed at "
                                       f"epoch={point[0]} batch={point[1]}: {exc}")
                if not keep_saving:
                    self.saving = False
                    self._cond.notify_all()
            raise
        with self._cond:
            if self.run_id == run_id:
                self.checkpoint_dirty = False
                self.last_error = None
                self._saved_point = point
                self.last_checkpoint_at = _now_iso()
            if not keep_saving:
                self.saving = False
                self._cond.notify_all()

    # ----------------------------------------------------------------- queries

    def status(self) -> dict:
        with self._cond:
            if self.cfg is not None:
                bpe = (self.num_samples + self.cfg.batch_size - 1) // self.cfg.batch_size
                total = self.cfg.epochs * bpe
                done = self.epoch * bpe + self.next_batch
            else:
                bpe = total = done = 0
            return {
                "state": self.state.value,
                "run_id": self.run_id,
                "recovered_after_restart": self.recovered,
                "epoch": self.epoch,
                "next_batch": self.next_batch,
                "batches_per_epoch": bpe,
                "total_batches": total,
                "completed_batches": done,
                "progress": (done / total) if total else 0.0,
                "saving": self.saving,
                "checkpoint_dirty": self.checkpoint_dirty,
                "last_checkpoint_at": self.last_checkpoint_at,
                "last_error": self.last_error,
                "started_at": self.started_at,
                "config": self.cfg.to_dict() if self.cfg else None,
                "dataset": {
                    "path": self.dataset_path,
                    "num_samples": self.num_samples,
                    "num_features": self.input_dim,
                    "num_classes": self.num_classes,
                },
            }

    def get_losses(self, since_step: int | None = None, limit: int | None = None) -> dict:
        with self._cond:
            losses = list(self.losses)
            total = len(losses)
        if since_step is not None:
            losses = [l for l in losses if l["step"] > since_step]
        if limit is not None:
            losses = losses[:limit]
        return {"count": total, "returned": len(losses), "losses": losses}

    def read_parameters(self) -> bytes:
        """Model parameters from the latest persisted checkpoint, as a .pt file."""
        ckpt = load_checkpoint(self.runs_dir)
        if ckpt is None:
            raise NotFoundError("no checkpoint exists yet; train first")
        buf = io.BytesIO()
        torch.save(ckpt["model_state_dict"], buf)
        return buf.getvalue()

    def checkpoint_file(self) -> Path:
        path = checkpoint_path(self.runs_dir)
        if not path.exists():
            raise NotFoundError("no checkpoint exists yet; train first")
        return path
