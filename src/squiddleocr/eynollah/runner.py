"""Run eynollah as a subprocess on a batch of images, with a resource plan, low priority and a RAM watchdog.

``EynollahRunner.run(images, xml_dir)`` writes ``<stem>.xml`` into ``xml_dir`` for every image that
does not have one yet (resume-safe, like eynollah's own skip-if-exists) and returns
``{stem: xml path}``. The subprocess is ``python -m squiddleocr.eynollah.launch``, started with the
plan's environment at ``nice 10`` (and best-effort idle I/O priority) in its own session, so it can
be killed as a group. While it runs, available RAM is polled: below the plan's margin a warning is
logged; when it keeps falling the run is stopped and restarted with one job fewer on the pages
that are still missing. eynollah's log goes to ``xml_dir/eynollah.log``; warnings and errors are
forwarded, the harmless ONNX Runtime ``GPU device discovery failed`` line is dropped.

Once eynollah logs ``All jobs done`` every XML is written; its interpreter then spends up to
4.5 s per model process on shutdown (each predictor polls its queue with that timeout before it
sees the stop flag), 15 to 25 s for a full-layout run. The runner waits ``shutdown_grace`` seconds
for a clean exit and then ends the process group itself.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .resources import GiB, Machine, Plan, Policy, detect_machine, plan as make_plan

Log = Callable[[str], None]
LOG_LINE = re.compile(r"^\d\d:\d\d:\d\d\.\d{3} (?P<level>[A-Z]+) (?P<name>\S+) - (?P<msg>.*)$")
NOISE = ("GPU device discovery failed", "/sys/class/drm/card")
EYNOLLAH_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")     # what eynollah's -di accepts
LOG_NAME = "eynollah.log"


@dataclass
class RunResult:
    xml: dict[str, Path]
    plan: Plan | None
    providers: dict[str, str]          # model -> provider eynollah reported
    attempts: int
    seconds: float


class EynollahRunner:
    def __init__(self, model_dir: Path, policy: Policy | None = None, log: Log | None = None,
                 warn: Log | None = None, progress: Callable[[int, int], None] | None = None,
                 poll_interval: float = 1.0, verbose: bool | None = None, shutdown_grace: float = 2.0):
        self.model_dir = Path(model_dir)
        self.shutdown_grace = shutdown_grace
        self.policy = policy or Policy()
        self.log = log or (lambda s: None)
        self.warn = warn or self.log
        self.progress = progress
        self.poll_interval = poll_interval
        self.verbose = bool(os.environ.get("SQUIDDLE_VERBOSE")) if verbose is None else verbose
        self.machine: Machine | None = None

    # ------------------------------------------------------------------ public
    def run(self, images: Sequence[Path], xml_dir: Path) -> RunResult:
        t0 = time.perf_counter()
        xml_dir = Path(xml_dir)
        xml_dir.mkdir(parents=True, exist_ok=True)
        by_stem = _by_stem(images)
        pending = {s: p for s, p in by_stem.items() if not _valid_xml(xml_dir / f"{s}.xml")}
        if not pending:
            return RunResult({s: xml_dir / f"{s}.xml" for s in by_stem}, None, {}, 0, 0.0)
        self.machine = self.machine or detect_machine()
        plan = make_plan(self.machine, _largest_pixels(pending.values()), self.policy)
        self.log(plan.describe())
        for note in plan.notes:
            self.warn(note)
        providers: dict[str, str] = {}
        attempts = 0
        jobs = plan.jobs
        while pending:
            attempts += 1
            outcome = self._attempt(pending, xml_dir, plan, jobs, providers, len(by_stem) - len(pending))
            pending = {s: p for s, p in pending.items() if not _valid_xml(xml_dir / f"{s}.xml")}
            if outcome == "restart" and pending:
                jobs = max(1, jobs - 1)
                self.warn(f"restarting eynollah with {jobs} job(s) on the {len(pending)} remaining page(s)")
                continue
            break
        if plan.use_gpu and providers and any(p.startswith("CPU") for p in providers.values()):
            self.warn("eynollah ran these models on the CPU: " + ", ".join(m for m, p in providers.items() if p.startswith("CPU")))
        return RunResult({s: xml_dir / f"{s}.xml" for s in by_stem if _valid_xml(xml_dir / f"{s}.xml")},
                         plan, providers, attempts, time.perf_counter() - t0)

    def command(self, in_dir: Path, xml_dir: Path, plan: Plan, jobs: int) -> list[str]:
        cmd = [sys.executable, "-m", "squiddleocr.eynollah.launch", "-m", str(self.model_dir), "-l", "INFO"]
        if plan.device:
            cmd += ["-D", plan.device]
        cmd += ["layout", "-di", str(in_dir), "-o", str(xml_dir), "-j", str(jobs)]
        cmd += list(self.policy.args)
        return cmd

    # ------------------------------------------------------------------ one subprocess
    def _attempt(self, pending: dict[str, Path], xml_dir: Path, plan: Plan, jobs: int,
                 providers: dict[str, str], done_before: int) -> str:
        """Run eynollah on ``pending``; returns ``done``, ``restart`` (killed by the watchdog) or ``failed``."""
        in_dir = xml_dir / ".squiddle_in"
        _stage_inputs(pending, in_dir)
        cmd = self.command(in_dir, xml_dir, plan, jobs)
        env = {**os.environ, **plan.env()}
        log_path = xml_dir / LOG_NAME
        with open(log_path, "a", encoding="utf-8") as logf:
            logf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {shlex.join(cmd)}\n=== {plan.describe()}\n")
            logf.flush()
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                                    bufsize=1, errors="replace", **_low_priority_kwargs())
            finished = threading.Event()
            reader = threading.Thread(target=self._pump, args=(proc, logf, providers, finished), daemon=True)
            reader.start()
            outcome = self._watch(proc, pending, xml_dir, plan, jobs, done_before, finished)
            reader.join(timeout=5)
        shutil.rmtree(in_dir, ignore_errors=True)
        if outcome == "failed":
            tail = _tail(log_path)
            raise RuntimeError(f"eynollah exited with status {proc.returncode}; see {log_path}\n{tail}")
        return outcome

    def _pump(self, proc: subprocess.Popen, logf, providers: dict[str, str], finished: threading.Event) -> None:
        in_traceback = False
        for line in proc.stderr:
            logf.write(line)
            line = line.rstrip("\n")
            if any(n in line for n in NOISE):
                continue
            m = LOG_LINE.match(line)
            if m:
                level, msg = m.group("level"), m.group("msg")
                in_traceback = level in ("ERROR", "CRITICAL", "FATAL")
                pm = re.search(r"ONNX provider (\S+) for model (\S+)", msg)
                if pm:
                    providers[pm.group(2)] = pm.group(1)
                if msg.startswith("All jobs done"):
                    finished.set()
                if self.verbose or level in ("WARNING", "ERROR", "CRITICAL", "FATAL"):
                    self.warn(f"eynollah: {msg}" if level == "WARNING" else f"eynollah {level}: {msg}")
            elif self.verbose or in_traceback or line.startswith(("Traceback", "[E:onnxruntime")):
                self.warn(line)
        logf.flush()

    def _watch(self, proc: subprocess.Popen, pending: dict[str, Path], xml_dir: Path, plan: Plan, jobs: int,
               done_before: int, finished: threading.Event | None = None) -> str:
        total = done_before + len(pending)
        below = 0
        warned = False
        last_available = None
        last_done = -1
        finished_at = None
        while True:
            rc = proc.poll()
            done = sum(1 for s in pending if (xml_dir / f"{s}.xml").is_file())
            if done != last_done and self.progress:
                self.progress(done_before + done, total)
                last_done = done
            if rc is not None:
                return "done" if rc == 0 or finished_at is not None else "failed"
            if finished is not None and finished.is_set():
                finished_at = finished_at or time.monotonic()
                if time.monotonic() - finished_at >= self.shutdown_grace:
                    _kill(proc, grace=self.shutdown_grace)     # all XML written; skip eynollah's slow teardown
                    return "done"
            available = _available_ram()
            if available is not None and plan.ram_margin:
                if available < plan.ram_margin:
                    below += 1
                    if not warned:
                        self.warn(f"available RAM {available / GiB:.1f} GB is below the margin of {plan.ram_margin / GiB:.1f} GB")
                        warned = True
                    falling = last_available is not None and available < last_available
                    if jobs > 1 and (available < plan.ram_margin / 2 or (below >= 3 and falling)):
                        self.warn(f"available RAM keeps falling ({available / GiB:.1f} GB): stopping eynollah")
                        _kill(proc)
                        _drop_partial_xml(pending, xml_dir)
                        return "restart"
                else:
                    below = 0
                last_available = available
            time.sleep(self.poll_interval)


# ---------------------------------------------------------------------- helpers
def _by_stem(images: Sequence[Path]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in images:
        p = Path(p)
        if p.stem in out and out[p.stem] != p:
            raise ValueError(f"two inputs share the stem {p.stem!r} ({out[p.stem]} and {p}); eynollah names its XML by stem")
        out[p.stem] = p
    return out


def _valid_xml(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        ET.parse(path)
        return True
    except ET.ParseError:
        return False


def _largest_pixels(paths) -> int:
    from PIL import Image

    largest = 0
    for p in paths:
        try:
            with Image.open(p) as im:
                largest = max(largest, im.width * im.height)
        except OSError:
            continue
    return largest


def _stage_inputs(pending: dict[str, Path], in_dir: Path) -> None:
    """A directory of links to the pending images for eynollah's ``-di``; formats it does not read
    (WebP, BMP) are written as PNG."""
    from PIL import Image

    shutil.rmtree(in_dir, ignore_errors=True)
    in_dir.mkdir(parents=True)
    for stem, src in pending.items():
        if src.suffix.lower() in EYNOLLAH_SUFFIXES:
            target = in_dir / f"{stem}{src.suffix.lower()}"
            try:
                os.symlink(src.resolve(), target)
            except OSError:
                shutil.copyfile(src, target)
        else:
            with Image.open(src) as im:
                im.convert("RGB").save(in_dir / f"{stem}.png")


def _low_priority_kwargs() -> dict:
    if os.name == "nt":  # pragma: no cover
        return {"creationflags": getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)}

    def preexec():
        try:
            os.nice(10)
        except OSError:
            pass
        try:
            import psutil

            psutil.Process().ionice(psutil.IOPRIO_CLASS_BE, 7)
        except Exception:  # noqa: BLE001 - best effort
            pass

    return {"preexec_fn": preexec, "start_new_session": True}


def _available_ram() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        try:
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        except OSError:
            pass
    return None


def _kill(proc: subprocess.Popen, grace: float = 10.0) -> None:
    """Terminate eynollah and every process it forked or spawned (its own session)."""
    try:
        os.killpg(proc.pid, signal.SIGTERM) if hasattr(os, "killpg") else proc.terminate()
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL) if hasattr(os, "killpg") else proc.kill()
        except ProcessLookupError:
            pass
        proc.wait(timeout=grace)


def _drop_partial_xml(pending: dict[str, Path], xml_dir: Path) -> None:
    for s in pending:
        p = xml_dir / f"{s}.xml"
        if p.exists() and not _valid_xml(p):
            p.unlink()


def _tail(path: Path, n: int = 15) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])
