"""The runner's watchdog and restart logic with a fake subprocess (eynollah itself never starts)."""
import threading
from pathlib import Path

from squiddleocr.eynollah import runner as runner_mod
from squiddleocr.eynollah.resources import GiB, Machine, Plan, Policy
from squiddleocr.eynollah.runner import EynollahRunner, RunResult


class FakeProc:
    def __init__(self):
        self.returncode = None
        self.pid = 4242
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


def _plan(jobs=4):
    return Plan(jobs=jobs, threads=2, reserved_cores=2, device="CPU", providers=["CPU"], use_gpu=False, models=(),
                vram_limits={}, ram_margin=8 * GiB)


def test_watchdog_warns_then_restarts_when_ram_keeps_falling(tmp_path, monkeypatch):
    samples = iter([20 * GiB, 7 * GiB, 6 * GiB, 5 * GiB, 4 * GiB, 3 * GiB])
    monkeypatch.setattr(runner_mod, "_available_ram", lambda: next(samples))
    proc = FakeProc()

    def fake_kill(p, grace=10.0):
        p.killed, p.returncode = True, -15

    monkeypatch.setattr(runner_mod, "_kill", fake_kill)
    warnings = []
    r = EynollahRunner(tmp_path, Policy(), warn=warnings.append, poll_interval=0.0)
    (tmp_path / "b.xml").write_text("<a>")                    # a half-written file from the killed run
    outcome = r._watch(proc, {"a": Path("a.jpg"), "b": Path("b.jpg")}, tmp_path, _plan(4), 4, 0)
    assert outcome == "restart" and proc.killed
    assert any("below the margin" in w for w in warnings) and any("keeps falling" in w for w in warnings)
    assert not (tmp_path / "b.xml").exists()                  # partial XML dropped before the retry


def test_watchdog_only_warns_with_a_single_job(tmp_path, monkeypatch):
    samples = iter([7 * GiB, 6 * GiB, 5 * GiB, 4 * GiB, 3 * GiB, 2 * GiB, 1 * GiB])
    proc = FakeProc()

    def sample():
        v = next(samples)
        if v <= 2 * GiB:
            proc.returncode = 0
        return v

    monkeypatch.setattr(runner_mod, "_available_ram", sample)
    warnings = []
    r = EynollahRunner(tmp_path, Policy(), warn=warnings.append, poll_interval=0.0)
    assert r._watch(proc, {"a": Path("a.jpg")}, tmp_path, _plan(1), 1, 0) == "done"
    assert sum("below the margin" in w for w in warnings) == 1 and not any("keeps falling" in w for w in warnings)


def test_watchdog_ends_the_process_after_all_jobs_done(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "_available_ram", lambda: 50 * GiB)
    proc = FakeProc()
    killed = []
    monkeypatch.setattr(runner_mod, "_kill", lambda p, grace=10.0: killed.append(grace))
    (tmp_path / "a.xml").write_text("<a/>")
    progress = []
    r = EynollahRunner(tmp_path, Policy(), progress=lambda d, t: progress.append((d, t)), poll_interval=0.0, shutdown_grace=0.0)
    finished = threading.Event()
    finished.set()
    assert r._watch(proc, {"a": Path("a.jpg")}, tmp_path, _plan(2), 2, 1, finished) == "done"
    assert killed == [0.0] and progress[-1] == (2, 2)


def test_run_retries_with_one_job_fewer_and_skips_existing_xml(tmp_path, monkeypatch):
    imgs = [tmp_path / "a.jpg", tmp_path / "b.jpg", tmp_path / "c.jpg"]
    (tmp_path / "c.xml").write_text("<a/>")                   # already done: never pending
    xml_dir = tmp_path / "xml"
    (tmp_path / "xml").mkdir()
    (xml_dir / "c.xml").write_text("<a/>")
    calls = []

    def fake_attempt(pending, xml_dir, plan, jobs, providers, done_before):
        calls.append((sorted(pending), jobs, done_before))
        if len(calls) == 1:
            (xml_dir / "a.xml").write_text("<a/>")
            return "restart"
        (xml_dir / "b.xml").write_text("<a/>")
        providers["page"] = "CUDA"
        return "done"

    r = EynollahRunner(tmp_path, Policy(jobs=3), log=lambda s: None, warn=lambda s: None)
    r.machine = Machine(8, 8, 32 * GiB, 24 * GiB)
    monkeypatch.setattr(r, "_attempt", fake_attempt)
    monkeypatch.setattr(runner_mod, "_largest_pixels", lambda paths: 1000)
    res = r.run(imgs, xml_dir)
    assert calls == [(["a", "b"], 3, 1), (["b"], 2, 2)]
    assert isinstance(res, RunResult) and res.attempts == 2 and sorted(res.xml) == ["a", "b", "c"]
    assert res.providers == {"page": "CUDA"}
    again = r.run(imgs, xml_dir)
    assert again.attempts == 0 and again.plan is None and len(calls) == 2


def test_stage_inputs_links_supported_and_converts_the_rest(tmp_path):
    from PIL import Image

    img = Image.new("RGB", (8, 8), "white")
    img.save(tmp_path / "a.jpg")
    img.save(tmp_path / "b.webp")
    in_dir = tmp_path / "in"
    runner_mod._stage_inputs({"a": tmp_path / "a.jpg", "b": tmp_path / "b.webp"}, in_dir)
    assert sorted(p.name for p in in_dir.iterdir()) == ["a.jpg", "b.png"]
    assert (in_dir / "a.jpg").is_symlink() or (in_dir / "a.jpg").stat().st_size == (tmp_path / "a.jpg").stat().st_size
