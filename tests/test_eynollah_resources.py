"""The eynollah resource policy on synthetic machine profiles (no psutil, GPU or eynollah needed)."""
import json

import pytest

from squiddleocr.eynollah.resources import (FALLBACK_VRAM_MB, GiB, MiB, Machine, Policy, parse_vram_margin, plan,
                                            resident_models)

PAGE = 4000 * 6000          # pixels of a large scan
DEFAULTS = FALLBACK_VRAM_MB


def laptop():
    return Machine(logical_cores=16, physical_cores=8, total_ram=32 * GiB, available_ram=24 * GiB)


def workstation():
    return Machine(logical_cores=64, physical_cores=32, total_ram=256 * GiB, available_ram=200 * GiB,
                   gpu_name="RTX 6000 Ada", gpu_total=24 * GiB, gpu_free=22 * GiB, cuda_provider=True,
                   tensorrt_provider=True, tensorrt_loadable=False)


def spark():
    return Machine(logical_cores=20, physical_cores=20, total_ram=128 * GiB, available_ram=115 * GiB,
                   gpu_name="NVIDIA GB10", gpu_total=128 * GiB, gpu_free=115 * GiB, unified=True, cuda_provider=True)


def test_resident_models_follow_the_flags():
    assert resident_models(("-fl", "-romb")) == ("col_classifier", "page", "textline", "region_1_2", "region_fl_np", "reading_order")
    assert resident_models(()) == ("col_classifier", "page", "textline", "region_1_2")
    assert "table" in resident_models(("-tab",)) and "binarization" in resident_models(("--input_binary",))


def test_cpu_only_laptop():
    p = plan(laptop(), PAGE, Policy(), DEFAULTS)
    assert not p.use_gpu and p.device == "CPU" and p.providers == ["CPU"] and p.vram_limits == {}
    assert p.reserved_cores == 2
    per_job = PAGE * 3 * 8                                   # 576 MB -> the 1.5 GB floor applies
    assert p.per_job_ram == int(1.5 * GiB) and per_job < p.per_job_ram
    budget = 24 * GiB - max(0.25 * 24 * GiB, 8 * GiB)      # 16 GB
    assert p.ram_budget == budget
    assert p.jobs == min(8 - 2, budget // p.per_job_ram, 8) == 6
    assert p.jobs * p.per_job_ram <= p.ram_budget
    assert p.threads == (8 - 2) // 6 == 1
    assert p.jobs * p.threads <= 8 - p.reserved_cores
    assert "no GPU" in " ".join(p.notes)
    assert "SQUIDDLE_EYNOLLAH_VRAM" not in p.env() and p.env()["EYNOLLAH_ONNX_EP"] == "CPU"


def test_discrete_gpu_workstation():
    p = plan(workstation(), PAGE, Policy(), DEFAULTS)
    assert p.use_gpu and p.device == "GPU" and p.providers == ["CUDA", "CPU"]
    assert p.gpu_margin == int(max(0.2 * 24 * GiB, 4 * GiB))
    remainder = 22 * GiB - p.gpu_margin - 2 * GiB          # 15.2 GB for six models
    total_w = sum(DEFAULTS[m] for m in p.models)
    for m in p.models:
        share = int(remainder * DEFAULTS[m] / total_w / MiB)
        assert p.vram_limits[m] == max(2 * DEFAULTS[m], min(share, 8 * 1024))
        assert 2 * DEFAULTS[m] <= p.vram_limits[m] <= 8 * 1024
    assert p.gpu_budget == sum(p.vram_limits.values()) * MiB <= remainder
    assert p.reserved_cores == max(2, 4) == 4
    assert p.ram_budget == 200 * GiB - 50 * GiB              # discrete GPU: RAM budget untouched by the caps
    assert p.jobs == 8 and p.threads == (32 - 4) // 8 == 3
    env = p.env()
    assert json.loads(env["SQUIDDLE_EYNOLLAH_VRAM"]) == p.vram_limits
    assert env["OMP_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "3"
    assert p.notes == []


def test_unified_memory_spark():
    p = plan(spark(), PAGE, Policy(), DEFAULTS)
    assert p.use_gpu and p.gpu_margin == int(0.2 * 128 * GiB)
    assert all(v == 8 * 1024 for m, v in p.vram_limits.items() if m != "col_classifier")   # the 8 GB ceiling
    assert p.vram_limits["col_classifier"] < 8 * 1024
    ram_after_margin = 115 * GiB - int(max(0.25 * 115 * GiB, 8 * GiB))
    assert p.ram_budget == ram_after_margin - p.gpu_budget   # one pool: the caps come out of the RAM budget
    assert p.jobs == 8 and p.threads == 2 and p.reserved_cores == 2
    assert p.jobs * p.per_job_ram <= p.ram_budget
    assert p.jobs * p.threads <= 20 - p.reserved_cores
    assert "unified with NVIDIA GB10" in p.describe() and "jobs 8 x 2 threads" in p.describe()


def test_overrides_and_small_gpu_notes():
    m = workstation()
    p = plan(m, PAGE, Policy(jobs=2, device="col*:CPU,*:GPU0", vram_margin=0.5, tensorrt=True), DEFAULTS)
    assert p.jobs == 2 and p.threads == (32 - 4) // 2 and p.device == "col*:CPU,*:GPU0"
    assert p.gpu_margin == 12 * GiB
    assert p.providers == ["CUDA", "CPU"] and any("libnvinfer" in n for n in p.notes)
    m.tensorrt_loadable = True
    assert plan(m, PAGE, Policy(tensorrt=True), DEFAULTS).providers == ["Tensorrt", "CUDA", "CPU"]
    assert plan(m, PAGE, Policy(vram_margin=6.0), DEFAULTS).gpu_margin == 6 * GiB
    assert plan(m, PAGE, Policy(device="CPU"), DEFAULTS).use_gpu is False
    small = Machine(8, 8, 32 * GiB, 24 * GiB, "GTX 1650", 4 * GiB, 3.5 * GiB, cuda_provider=True)
    p = plan(small, PAGE, Policy(), DEFAULTS)
    assert p.use_gpu and all(v == 2 * DEFAULTS[k] for k, v in p.vram_limits.items())   # the floors
    assert any("below the caps' floors" in n for n in p.notes)
    no_cuda = Machine(8, 8, 32 * GiB, 24 * GiB, "RTX", 8 * GiB, 8 * GiB, cuda_provider=False)
    p = plan(no_cuda, PAGE, Policy(), DEFAULTS)
    assert not p.use_gpu and any("CUDAExecutionProvider" in n for n in p.notes)


def test_jobs_never_below_one_and_big_pages_count():
    tiny = Machine(2, 2, 4 * GiB, 3 * GiB)
    p = plan(tiny, 8000 * 12000, Policy(), DEFAULTS)
    assert p.jobs == 1 and p.threads == 1 and p.per_job_ram == 8000 * 12000 * 24
    assert any("exceed the RAM budget" in n for n in p.notes)


def test_parse_vram_margin():
    assert parse_vram_margin(None) is None and parse_vram_margin("") is None
    assert parse_vram_margin("0.2") == 0.2 and parse_vram_margin("20%") == 0.2
    assert parse_vram_margin("4") == 4.0 and parse_vram_margin("4G") == 4.0 and parse_vram_margin("6gb") == 6.0
    with pytest.raises(ValueError):
        parse_vram_margin("lots")
