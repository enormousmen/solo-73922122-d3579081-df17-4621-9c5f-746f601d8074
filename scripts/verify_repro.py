#!/usr/bin/env python3
"""End-to-end verification of bit-exact resumable training.

Spins up real uvicorn server processes and checks, against an uninterrupted
baseline (identical seed + config):

  A   uninterrupted baseline run (reference)
  B   pause + resume                  -> losses & final params bit-identical to A
  C   hard crash + restart + resume   -> bit-identical to A
  D   two concurrent resumes          -> exactly one 200, one 409; result == A
  E1  checkpoint save failure         -> pause reports 500, old checkpoint kept,
                                         retry succeeds, resume reproduces A
  E2  save failure, then hard crash   -> restart recovers the OLD checkpoint,
                                         resume still reproduces A
  F   slow checkpoint saves           -> status queries stay fast while saving
  G   hard crash in the middle of a save -> restart finds a complete (old or
                                         new) checkpoint, resume reproduces
                                         G-baseline
  H   service basics                  -> /, /docs, 404s before first checkpoint

Exit code 0 iff every scenario passes.  Run from the repo root:

    python3 scripts/verify_repro.py
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

# Shared by scenarios A/B/C/D/E (bit-exact comparison target).
CFG_MAIN = {"epochs": 5, "batch_size": 32, "batch_sleep_seconds": 0.03,
            "checkpoint_every": 2, "checkpoint_sleep_seconds": 0.0}
# Scenario F: slow saves, short run.
CFG_SLOW_SAVE = {"epochs": 1, "batch_size": 32, "batch_sleep_seconds": 0.03,
                 "checkpoint_every": 1, "checkpoint_sleep_seconds": 1.0}
# Scenario G: kill mid-save, short run.
CFG_KILL_SAVE = {"epochs": 1, "batch_size": 32, "batch_sleep_seconds": 0.03,
                 "checkpoint_every": 1, "checkpoint_sleep_seconds": 0.5}

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, note: str = "") -> None:
    RESULTS.append((name, ok, note))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {note}" if note else ""), flush=True)


# --------------------------------------------------------------------- http

def api(port: int, method: str, path: str, body: dict | None = None,
        timeout: float = 120.0) -> tuple[int, dict]:
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"detail": raw.decode(errors="replace")}


def api_bytes(port: int, path: str) -> bytes:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=60) as resp:
        return resp.read()


# ------------------------------------------------------------------ servers

class Server:
    def __init__(self, runs_dir: Path, log_dir: Path, tag: str):
        self.runs_dir = runs_dir
        self.port = _free_port()
        self.tag = tag
        log = open(log_dir / f"server-{tag}.log", "wb")
        env = dict(os.environ)
        env["RUNS_DIR"] = str(runs_dir)
        env["CONFIG_PATH"] = str(ROOT / "config" / "default.json")
        env["ENABLE_TEST_CRASH_ENDPOINT"] = "1"
        self.proc = subprocess.Popen(
            [PY, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
             "--port", str(self.port)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        self._log_path = log_dir / f"server-{tag}.log"
        self._wait_ready()

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"server {self.tag} exited early; log:\n"
                    + self._log_path.read_text(errors="replace")[-3000:])
            try:
                code, _ = api(self.port, "GET", "/train/status", timeout=2)
                if code == 200:
                    return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(0.3)
        raise RuntimeError(f"server {self.tag} did not become ready")

    def kill(self) -> None:
        """Sudden death via the test-only crash endpoint (os._exit).

        Equivalent to SIGKILL for recovery purposes: the process dies
        immediately, wherever it is (including mid-checkpoint-save), with no
        cleanup.  (Real SIGKILL cannot be used in every environment; Docker
        users can equally use `docker kill`, see README.)
        """
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/test/crash", data=b"", timeout=5)
        except Exception:
            pass  # the connection drops when the process dies
        self.proc.wait(timeout=10)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.kill()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ------------------------------------------------------------------ helpers

def wait_state(port: int, want: str, timeout: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code, st = api(port, "GET", "/train/status")
        if code == 200 and st["state"] == want:
            return st
        time.sleep(0.2)
    raise RuntimeError(f"state did not become {want!r} within {timeout}s")


def wait_until(port: int, pred, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code, st = api(port, "GET", "/train/status")
        if code == 200 and pred(st):
            return st
        time.sleep(0.05)
    raise RuntimeError("condition not met within timeout")


def start_run(port: int, cfg: dict) -> dict:
    code, st = api(port, "POST", "/train/start", {"force": True, "config": cfg})
    assert code == 200, f"start failed: {st}"
    return st


def run_to_finish(port: int, cfg: dict) -> tuple[list, bytes]:
    start_run(port, cfg)
    st = wait_state(port, "finished")
    assert st["completed_batches"] == st["total_batches"], st
    _, losses = api(port, "GET", "/train/losses")
    params = api_bytes(port, "/train/parameters")
    return losses["losses"], params


def compare_to_baseline(name: str, ref_losses: list, ref_params: bytes,
                        got_losses: list, got_params: bytes) -> bool:
    ok = True
    if got_losses == ref_losses:
        print(f"    losses: {len(got_losses)} entries, exact match")
    else:
        ok = False
        n = min(len(got_losses), len(ref_losses))
        diff = next((i for i in range(n) if got_losses[i] != ref_losses[i]), None)
        print(f"    losses: MISMATCH (len {len(got_losses)} vs {len(ref_losses)}, "
              f"first diff at index {diff})")
        if diff is not None:
            print(f"      got: {got_losses[diff]}\n      ref: {ref_losses[diff]}")
    ref_sd = torch.load(io.BytesIO(ref_params), weights_only=True)
    got_sd = torch.load(io.BytesIO(got_params), weights_only=True)
    if ref_sd.keys() != got_sd.keys():
        ok = False
        print(f"    params: key mismatch {sorted(got_sd)} vs {sorted(ref_sd)}")
    elif all(torch.equal(got_sd[k], ref_sd[k]) for k in ref_sd):
        h_ref = hashlib.sha256(ref_params).hexdigest()[:16]
        h_got = hashlib.sha256(got_params).hexdigest()[:16]
        print(f"    params: all tensors bit-identical "
              f"(sha256 {h_got}{' ==' if h_got == h_ref else ' !='} {h_ref})")
    else:
        ok = False
        bad = [k for k in ref_sd if not torch.equal(got_sd[k], ref_sd[k])]
        print(f"    params: MISMATCH in tensors {bad}")
    record(name, ok)
    return ok


# ------------------------------------------------------------------ scenarios

def scenario_basics(port: int) -> None:
    print("\n== H: service basics (idle server) ==")
    code, _ = api(port, "GET", "/")
    record("H1 GET /", code == 200)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/docs",
                                    timeout=10) as resp:
            docs_ok = resp.status == 200 and b"swagger" in resp.read().lower()
    except Exception:
        docs_ok = False
    record("H2 GET /docs (Swagger UI)", docs_ok)
    code, _ = api(port, "GET", "/train/parameters")
    record("H3 parameters before any checkpoint -> 404", code == 404)
    code, _ = api(port, "POST", "/train/resume")
    record("H4 resume with no run -> 404", code == 404)
    code, _ = api(port, "POST", "/train/pause")
    record("H5 pause while idle -> 409", code == 409)


def scenario_pause_resume(port: int, ref: tuple[list, bytes]) -> None:
    print("\n== B: pause + resume ==")
    start_run(port, CFG_MAIN)
    wait_until(port, lambda s: s["completed_batches"] >= 12)
    t0 = time.monotonic()
    code, st = api(port, "POST", "/train/pause")
    dt = time.monotonic() - t0
    ok = (code == 200 and st["state"] == "paused" and not st["checkpoint_dirty"]
          and st["last_checkpoint_at"] is not None)
    record("B1 pause returns only after checkpoint persisted", ok,
           f"pause latency {dt:.2f}s, stopped at batch {st['completed_batches']}")
    code, st = api(port, "POST", "/train/resume")
    record("B2 resume accepted", code == 200 and st["state"] == "running")
    wait_state(port, "finished")
    _, losses = api(port, "GET", "/train/losses")
    params = api_bytes(port, "/train/parameters")
    compare_to_baseline("B3 pause+resume == uninterrupted", ref[0], ref[1],
                        losses["losses"], params)


def scenario_concurrent_resume(port: int, ref: tuple[list, bytes]) -> None:
    print("\n== D: concurrent resume starts exactly one trainer ==")
    start_run(port, CFG_MAIN)
    wait_until(port, lambda s: s["completed_batches"] >= 8)
    api(port, "POST", "/train/pause")
    codes: list[int] = []

    def fire():
        code, _ = api(port, "POST", "/train/resume")
        codes.append(code)

    threads = [threading.Thread(target=fire) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    record("D1 concurrent resumes -> one 200, one 409",
           sorted(codes) == [200, 409], f"got {sorted(codes)}")
    wait_state(port, "finished")
    _, losses = api(port, "GET", "/train/losses")
    params = api_bytes(port, "/train/parameters")
    compare_to_baseline("D2 resumed run == uninterrupted", ref[0], ref[1],
                        losses["losses"], params)


def scenario_save_failure_retry(port: int, runs_dir: Path,
                                ref: tuple[list, bytes]) -> None:
    print("\n== E1: checkpoint save failure keeps old checkpoint, retry works ==")
    start_run(port, CFG_MAIN)
    wait_until(port, lambda s: s["completed_batches"] >= 6
               and s["last_checkpoint_at"] is not None)
    flag = runs_dir / "INJECT_SAVE_FAILURES"
    flag.write_text("1")                      # arm test-only fault injection
    time.sleep(0.3)                           # let any in-flight save finish
    ckpt = runs_dir / "checkpoint.pt"
    hash_before = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    code, body = api(port, "POST", "/train/pause")
    record("E1 pause with failing save -> 500", code == 500,
           str(body.get("detail", ""))[:100])
    code, st = api(port, "GET", "/train/status")
    record("E1 status shows dirty checkpoint + error",
           st["state"] == "paused" and st["checkpoint_dirty"]
           and "injected fault" in (st["last_error"] or ""))
    hash_after = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    record("E1 old checkpoint untouched", hash_before == hash_after)
    flag.unlink()                             # fix the "storage"
    code, st = api(port, "POST", "/train/pause")
    record("E1 retried pause -> 200, checkpoint persisted",
           code == 200 and not st["checkpoint_dirty"])
    code, _ = api(port, "POST", "/train/resume")
    record("E1 resume after retry", code == 200)
    wait_state(port, "finished")
    _, losses = api(port, "GET", "/train/losses")
    params = api_bytes(port, "/train/parameters")
    compare_to_baseline("E1 run after save-failure == uninterrupted",
                        ref[0], ref[1], losses["losses"], params)


def scenario_slow_save_queries(port: int) -> None:
    print("\n== F: slow checkpoint saves do not block queries ==")
    start_run(port, CFG_SLOW_SAVE)
    wait_until(port, lambda s: s["saving"], timeout=30)
    lat = []
    for _ in range(3):
        t0 = time.monotonic()
        code, _ = api(port, "GET", "/train/status")
        lat.append(time.monotonic() - t0)
    code_l, _ = api(port, "GET", "/train/losses")
    worst = max(lat)
    record("F1 queries respond fast while a 1s save is in flight",
           worst < 1.0 and code == 200 and code_l == 200,
           f"worst status latency {worst*1000:.0f}ms")
    code, st = api(port, "POST", "/train/pause", timeout=60)
    record("F2 pause during slow saves", code == 200 and st["state"] == "paused")
    api(port, "POST", "/train/resume")
    st = wait_state(port, "finished", timeout=120)
    record("F3 slow-save run finishes", st["state"] == "finished")


def scenario_kill_restart(runs_dir: Path, log_dir: Path, server: Server,
                          ref: tuple[list, bytes]) -> Server:
    print("\n== C: hard crash + restart recovers from latest checkpoint ==")
    start_run(server.port, CFG_MAIN)
    wait_until(server.port, lambda s: s["completed_batches"] >= 10
               and s["last_checkpoint_at"] is not None)
    server.kill()
    record("C1 server crashed mid-run", True)
    server2 = Server(runs_dir, log_dir, "C-restarted")
    code, st = api(server2.port, "GET", "/train/status")
    record("C2 restart recovers paused state from checkpoint",
           st["state"] == "paused" and st["recovered_after_restart"]
           and 0 < st["completed_batches"] < st["total_batches"],
           f"recovered at batch {st['completed_batches']}/{st['total_batches']}")
    code, _ = api(server2.port, "POST", "/train/resume")
    record("C3 resume after restart", code == 200)
    wait_state(server2.port, "finished")
    _, losses = api(server2.port, "GET", "/train/losses")
    params = api_bytes(server2.port, "/train/parameters")
    compare_to_baseline("C4 kill+restart == uninterrupted", ref[0], ref[1],
                        losses["losses"], params)
    return server2


def scenario_save_failure_then_kill(runs_dir: Path, log_dir: Path, server: Server,
                                    ref: tuple[list, bytes]) -> Server:
    print("\n== E2: save failure, then hard crash -> old checkpoint used ==")
    start_run(server.port, CFG_MAIN)
    wait_until(server.port, lambda s: s["completed_batches"] >= 6
               and s["last_checkpoint_at"] is not None)
    flag = runs_dir / "INJECT_SAVE_FAILURES"
    flag.write_text("1")
    time.sleep(0.3)
    code, _ = api(server.port, "POST", "/train/pause")
    record("E2 pause with failing save -> 500", code == 500)
    server.kill()                              # die while dirty
    flag.unlink()
    server2 = Server(runs_dir, log_dir, "E2-restarted")
    code, st = api(server2.port, "GET", "/train/status")
    record("E2 restart falls back to old complete checkpoint",
           st["state"] == "paused" and st["recovered_after_restart"]
           and not st["checkpoint_dirty"],
           f"recovered at batch {st['completed_batches']}/{st['total_batches']}")
    code, _ = api(server2.port, "POST", "/train/resume")
    record("E2 resume after restart", code == 200)
    wait_state(server2.port, "finished")
    _, losses = api(server2.port, "GET", "/train/losses")
    params = api_bytes(server2.port, "/train/parameters")
    compare_to_baseline("E2 save-fail+kill == uninterrupted", ref[0], ref[1],
                        losses["losses"], params)
    return server2


def scenario_kill_mid_save(runs_dir: Path, log_dir: Path, server: Server) -> Server:
    print("\n== G: hard crash in the middle of a checkpoint save ==")
    print("  (baseline: identical config, uninterrupted)")
    ref_losses, ref_params = run_to_finish(server.port, CFG_KILL_SAVE)
    start_run(server.port, CFG_KILL_SAVE)
    # Crash while a save is in flight AND an earlier checkpoint of this run
    # already exists on disk (completed_batches >= 2 means the save for
    # batch 1 finished and the save for batch >= 2 is now running).
    wait_until(server.port,
               lambda s: s["saving"] and s["completed_batches"] >= 2, timeout=30)
    server.kill()                              # crash inside the save window
    record("G1 server crashed during save", True)
    server2 = Server(runs_dir, log_dir, "G-restarted")
    leftovers = [p.name for p in runs_dir.glob(".checkpoint.pt.tmp.*")]
    record("G2 no torn checkpoint / stale temp cleaned", leftovers == [],
           f"leftovers: {leftovers or 'none'}")
    code, st = api(server2.port, "GET", "/train/status")
    record("G3 restart recovers a complete checkpoint (old or new)",
           st["state"] == "paused" and st["recovered_after_restart"],
           f"recovered at batch {st['completed_batches']}/{st['total_batches']}")
    code, _ = api(server2.port, "POST", "/train/resume")
    record("G3b resume after restart", code == 200)
    wait_state(server2.port, "finished", timeout=120)
    _, losses = api(server2.port, "GET", "/train/losses")
    params = api_bytes(server2.port, "/train/parameters")
    compare_to_baseline("G4 kill-mid-save == uninterrupted", ref_losses, ref_params,
                        losses["losses"], params)
    return server2


# ---------------------------------------------------------------------- main

def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="verify-repro-"))
    runs_dir = work / "runs"
    log_dir = work / "logs"
    log_dir.mkdir()
    print(f"workdir: {work}")
    server = Server(runs_dir, log_dir, "main")
    try:
        scenario_basics(server.port)

        print("\n== A: uninterrupted baseline ==")
        ref_losses, ref_params = run_to_finish(server.port, CFG_MAIN)
        record("A baseline finished", True,
               f"{len(ref_losses)} batches, final loss {ref_losses[-1]['loss']:.6f}")

        scenario_pause_resume(server.port, (ref_losses, ref_params))
        scenario_concurrent_resume(server.port, (ref_losses, ref_params))
        scenario_save_failure_retry(server.port, runs_dir, (ref_losses, ref_params))
        scenario_slow_save_queries(server.port)

        server = scenario_kill_restart(runs_dir, log_dir, server,
                                       (ref_losses, ref_params))
        server = scenario_save_failure_then_kill(runs_dir, log_dir, server,
                                                 (ref_losses, ref_params))
        server = scenario_kill_mid_save(runs_dir, log_dir, server)
    finally:
        server.stop()

    print("\n================ SUMMARY ================")
    failed = 0
    for name, ok, note in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        failed += 0 if ok else 1
    print(f"  {len(RESULTS) - failed}/{len(RESULTS)} checks passed")
    if failed:
        print(f"  server logs under {log_dir}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
