"""What the machine has and how much of it an eynollah run may take.

``detect_machine`` reads cores (affinity and cgroup aware), RAM, the GPU (torch, else nvidia-smi)
and the ONNX Runtime providers; ``plan`` turns that and the largest input image into a ``Plan``:
the number of parallel page jobs, threads per job, the device spec, the provider list and the
per-model ONNX arena caps that replace eynollah's hard-coded ``MODEL_VRAM_LIMITS``. Everything is
pure arithmetic over a ``Machine`` so it can be tested on synthetic profiles.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field

GiB = 2**30
MiB = 2**20

#: eynollah 0.9.2's ``MODEL_VRAM_LIMITS`` (MB per model, calibrated for small discrete GPUs); the installed
#: eynollah's dict is preferred when importable. The ratios weight the split of the GPU budget.
FALLBACK_VRAM_MB = {"binarization": 868, "enhancement": 980, "col_classifier": 210, "page": 618, "textline": 1880,
                    "region_1_2": 1580, "region_fl_np": 1756, "table": 1818, "reading_order": 632, "ocr": 2600}
#: Models a ``layout`` run keeps resident (one worker process each): the base set plus one per flag.
BASE_MODELS = ("col_classifier", "page", "textline", "region_1_2")
FLAG_MODELS = {"-fl": "region_fl_np", "--full-layout": "region_fl_np",
               "-romb": "reading_order", "--reading_order_machine_based": "reading_order",
               "-tab": "table", "--tables": "table",
               "-ib": "binarization", "--input_binary": "binarization",
               "-ae": "enhancement", "--allow-enhancement": "enhancement"}
UNIFIED_GPU_NAMES = re.compile(r"GB10|GB200|Grace|Orin|Xavier|Thor|Jetson|Tegra|Spark", re.I)


def default_vram_limits() -> dict[str, int]:
    try:
        from eynollah.model_zoo.model_zoo import MODEL_VRAM_LIMITS

        return dict(MODEL_VRAM_LIMITS)
    except ImportError:
        return dict(FALLBACK_VRAM_MB)


@dataclass
class Machine:
    """A machine profile in bytes and cores. ``unified`` means the GPU shares system RAM (DGX Spark,
    Jetson): then ``gpu_total``/``gpu_free`` are the RAM figures and one budget covers both."""

    logical_cores: int
    physical_cores: int
    total_ram: int
    available_ram: int
    gpu_name: str = ""
    gpu_total: int = 0
    gpu_free: int = 0
    unified: bool = False
    cuda_provider: bool = False
    tensorrt_provider: bool = False
    tensorrt_loadable: bool = False

    @property
    def has_gpu(self) -> bool:
        return self.gpu_total > 0


def detect_machine() -> Machine:
    logical, physical = _cores()
    total, available = _memory()
    name, gpu_total, gpu_free = _gpu()
    unified = gpu_total > 0 and (gpu_total >= 0.8 * total or bool(UNIFIED_GPU_NAMES.search(name)) or _soc_signature())
    if unified:
        gpu_total, gpu_free = total, available      # one pool; psutil's "available" counts reclaimable cache, CUDA's does not
    providers = _providers()
    return Machine(logical, physical, total, available, name, gpu_total, gpu_free, unified,
                   cuda_provider="CUDAExecutionProvider" in providers,
                   tensorrt_provider="TensorrtExecutionProvider" in providers,
                   tensorrt_loadable=tensorrt_loadable())


def _cores() -> tuple[int, int]:
    try:
        logical = len(os.sched_getaffinity(0))
    except AttributeError:      # not on Linux
        logical = os.cpu_count() or 1
    physical = None
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    physical = min(physical or logical, logical)
    quota = _cgroup_cpu_quota()
    if quota:
        logical = max(1, min(logical, quota))
        physical = max(1, min(physical, quota))
    return logical, physical


def _cgroup_cpu_quota() -> int | None:
    """Whole CPUs allowed by a cgroup v2 ``cpu.max`` (or v1 cfs quota), None when unlimited/absent."""
    try:
        text = open("/sys/fs/cgroup/cpu.max").read().split()
        if text and text[0] != "max":
            return math.ceil(int(text[0]) / int(text[1]))
    except (OSError, ValueError, IndexError):
        pass
    try:
        quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if quota > 0 and period > 0:
            return math.ceil(quota / period)
    except (OSError, ValueError):
        pass
    return None


def _memory() -> tuple[int, int]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        total, available = int(vm.total), int(vm.available)
    except ImportError:
        info = {}
        try:
            for line in open("/proc/meminfo"):
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) * 1024
        except OSError:
            pass
        total = info.get("MemTotal", 8 * GiB)
        available = info.get("MemAvailable", total // 2)
    limit, used = _cgroup_memory()
    if limit and limit < total:
        total = limit
        available = max(0, min(available, limit - (used or 0)))
    return total, available


def _cgroup_memory() -> tuple[int | None, int | None]:
    for limit_file, used_file in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                                  ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            text = open(limit_file).read().strip()
            if text == "max":
                return None, None
            limit = int(text)
            if limit >= 1 << 60:        # v1 "unlimited"
                return None, None
            try:
                used = int(open(used_file).read())
            except (OSError, ValueError):
                used = None
            return limit, used
        except (OSError, ValueError):
            continue
    return None, None


def _gpu() -> tuple[str, int, int]:
    """(name, total, free) of GPU 0 in bytes; torch first (CUDA is already initialised by kraken in the
    parent process), else nvidia-smi; (\"\", 0, 0) without a GPU."""
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return torch.cuda.get_device_name(0), int(total), int(free)
    except Exception:  # noqa: BLE001 - any CUDA trouble means "no GPU through torch"
        pass
    try:
        import subprocess

        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        if out:
            name, total, used = [s.strip() for s in out[0].split(",")]
            if total.isdigit():
                total_b, used_b = int(total) * MiB, int(used) * MiB if used.isdigit() else 0
                return name, total_b, max(total_b - used_b, 0)
            return name, 0, 0
    except Exception:  # noqa: BLE001
        pass
    return "", 0, 0


def _soc_signature() -> bool:
    try:
        return "nvidia" in open("/proc/device-tree/model", "rb").read().decode("ascii", "ignore").lower()
    except OSError:
        return False


def _providers() -> list[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except ImportError:
        return []


def tensorrt_loadable() -> bool:
    """Whether ``libnvinfer`` can be loaded: without it ONNX Runtime lists the TensorRT provider and then
    silently runs on CPU."""
    import ctypes
    import ctypes.util

    for name in ("nvinfer", "libnvinfer.so.10", "libnvinfer.so", "nvinfer_10.dll", "nvinfer.dll"):
        try:
            ctypes.CDLL(ctypes.util.find_library(name) or name)
            return True
        except OSError:
            continue
    return False


# ------------------------------------------------------------------------------------------------ policy
@dataclass
class Policy:
    """Knobs of the allocation, all overridable from the CLI. ``jobs`` and ``device`` None = decide;
    ``vram_margin`` a fraction of GPU memory (<= 1) or gigabytes (> 1), None = max(20 %, 4 GB)."""

    jobs: int | None = None
    device: str | None = None
    vram_margin: float | None = None
    tensorrt: bool = False
    args: tuple[str, ...] = ("-fl", "-romb")
    kraken_reserve: int = 2 * GiB          # headroom for kraken's torch model in the same process family
    hard_cap: int = 8                      # jobs, whatever the machine says
    per_job_min: int = int(1.5 * GiB)
    per_job_copies: int = 8                # full-resolution working copies per page job (3 bytes/pixel)
    cap_floor_factor: float = 2.0          # per-model cap >= this x eynollah's default
    cap_ceiling: int = 8 * GiB


@dataclass
class Plan:
    """The allocation for one run: what the launcher applies and what the status line shows."""

    jobs: int
    threads: int
    reserved_cores: int
    device: str | None                     # eynollah -D spec, None = eynollah's own choice
    providers: list[str]                   # EYNOLLAH_ONNX_EP order
    use_gpu: bool
    models: tuple[str, ...]
    vram_limits: dict[str, int]            # MB per model, the patched MODEL_VRAM_LIMITS (empty on CPU)
    gpu_budget: int = 0                    # sum of the caps, bytes
    gpu_margin: int = 0
    per_job_ram: int = 0
    ram_budget: int = 0
    ram_margin: int = 0
    machine: Machine | None = None
    notes: list[str] = field(default_factory=list)

    def env(self) -> dict[str, str]:
        env = {"OMP_NUM_THREADS": str(self.threads), "OPENBLAS_NUM_THREADS": str(self.threads),
               "MKL_NUM_THREADS": str(self.threads), "SQUIDDLE_EYNOLLAH_THREADS": str(self.threads),
               "EYNOLLAH_ONNX_EP": ",".join(self.providers)}
        if self.vram_limits:
            env["SQUIDDLE_EYNOLLAH_VRAM"] = json.dumps(self.vram_limits)
        return env

    def describe(self) -> str:
        m = self.machine
        parts = []
        if m:
            parts.append(f"{m.physical_cores} cores ({self.reserved_cores} reserved)")
            ram = f"RAM {m.available_ram / GiB:.0f} of {m.total_ram / GiB:.0f} GB free"
            if m.has_gpu:
                ram += f", {'unified with' if m.unified else 'GPU'} {m.gpu_name or 'GPU'}"
                if not m.unified:
                    ram += f" {m.gpu_free / GiB:.0f} of {m.gpu_total / GiB:.0f} GB free"
            parts.append(ram)
        parts.append(f"jobs {self.jobs} x {self.threads} threads ({self.per_job_ram / GiB:.1f} GB/job, budget {self.ram_budget / GiB:.0f} GB)")
        if self.use_gpu:
            caps = " ".join(f"{k} {v / 1024:.1f}G" for k, v in self.vram_limits.items())
            parts.append(f"ONNX {'/'.join(self.providers)}, caps {caps} (sum {self.gpu_budget / GiB:.0f} GB, margin {self.gpu_margin / GiB:.0f} GB)")
        else:
            parts.append("ONNX CPU")
        return "  ·  ".join(parts)


def resident_models(args: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    models = list(BASE_MODELS)
    for a in args:
        m = FLAG_MODELS.get(a)
        if m and m not in models:
            models.append(m)
    return tuple(models)


def parse_vram_margin(text: str | None) -> float | None:
    """``0.2`` / ``20%`` = fraction of GPU memory; ``4`` / ``4G`` / ``4GB`` = gigabytes."""
    if text is None or str(text).strip() == "":
        return None
    s = str(text).strip().lower()
    if s.endswith("%"):
        return float(s[:-1]) / 100
    s = re.sub(r"(gb|gib|g)$", "", s)
    return float(s)


def plan(machine: Machine, largest_pixels: int, policy: Policy | None = None,
         defaults: dict[str, int] | None = None) -> Plan:
    """The allocation for ``machine``, pages of at most ``largest_pixels`` and the run's flags."""
    policy = policy or Policy()
    defaults = defaults or default_vram_limits()
    models = resident_models(policy.args)
    notes: list[str] = []

    # --- device and providers
    cpu_only = (policy.device or "").strip().upper() == "CPU"
    use_gpu = machine.has_gpu and machine.cuda_provider and not cpu_only
    if machine.has_gpu and not machine.cuda_provider and not cpu_only:
        notes.append("no CUDAExecutionProvider in this ONNX Runtime build: eynollah runs on the CPU")
    elif not machine.has_gpu and not cpu_only:
        notes.append("no GPU found: eynollah runs on the CPU")
    if use_gpu:
        providers = ["CUDA", "CPU"]
        if policy.tensorrt:
            if machine.tensorrt_provider and machine.tensorrt_loadable:
                providers.insert(0, "Tensorrt")
            else:
                notes.append("--eynollah-tensorrt ignored: " + ("libnvinfer is not loadable" if machine.tensorrt_provider
                                                                else "no TensorrtExecutionProvider in this ONNX Runtime build"))
        device = policy.device or "GPU"
    else:
        providers = ["CPU"]
        device = "CPU"

    # --- VRAM caps: split the free memory after margins across the resident models by their default ratios
    vram_limits: dict[str, int] = {}
    gpu_budget = gpu_margin = 0
    if use_gpu:
        if policy.vram_margin is None:
            gpu_margin = int(max(0.20 * machine.gpu_total, 4 * GiB))
        elif policy.vram_margin <= 1:
            gpu_margin = int(policy.vram_margin * machine.gpu_total)
        else:
            gpu_margin = int(policy.vram_margin * GiB)
        remainder = machine.gpu_free - gpu_margin - policy.kraken_reserve
        weights = {m: defaults.get(m, 1000) for m in models}
        total_w = sum(weights.values())
        for m in models:
            floor = int(policy.cap_floor_factor * defaults.get(m, 1000))
            ceiling = policy.cap_ceiling // MiB
            share = int(remainder * weights[m] / total_w / MiB) if remainder > 0 else 0
            vram_limits[m] = max(floor, min(share, ceiling))
        gpu_budget = sum(vram_limits.values()) * MiB
        if remainder < gpu_budget:
            notes.append(f"GPU memory after margins ({max(remainder, 0) / GiB:.1f} GB) is below the caps' floors "
                         f"({gpu_budget / GiB:.1f} GB); expect ONNX Runtime arena errors on a small GPU")

    # --- RAM budget and jobs
    ram_margin = int(max(0.25 * machine.available_ram, 8 * GiB))
    ram_budget = machine.available_ram - ram_margin
    if use_gpu and machine.unified:
        ram_budget -= gpu_budget
    ram_budget = max(ram_budget, 0)
    per_job = max(policy.per_job_min, int(largest_pixels) * 3 * policy.per_job_copies)
    reserved = max(2, math.ceil(0.10 * machine.physical_cores))
    usable = max(1, machine.physical_cores - reserved)
    auto_jobs = max(1, min(usable, ram_budget // per_job, policy.hard_cap))
    jobs = policy.jobs if policy.jobs and policy.jobs > 0 else auto_jobs
    if jobs * per_job > ram_budget:
        notes.append(f"{jobs} jobs x {per_job / GiB:.1f} GB exceed the RAM budget of {ram_budget / GiB:.1f} GB"
                     + ("" if policy.jobs else " (largest page is big and RAM is tight)"))
    threads = max(1, usable // jobs)
    return Plan(jobs=jobs, threads=threads, reserved_cores=reserved, device=device, providers=providers, use_gpu=use_gpu,
                models=models, vram_limits=vram_limits, gpu_budget=gpu_budget, gpu_margin=gpu_margin, per_job_ram=per_job,
                ram_budget=ram_budget, ram_margin=ram_margin, machine=machine, notes=notes)
