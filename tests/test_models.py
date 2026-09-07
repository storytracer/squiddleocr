import pytest

from squiddleocr import models


def test_resolve_path_passthrough(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        models.resolve_model(d)
    (d / "inference.onnx").write_bytes(b"x")
    (d / "inference.yml").write_text("Global: {}")
    assert models.resolve_model(d) == d


def test_resolve_uses_cache_then_download(tmp_path, monkeypatch):
    monkeypatch.setenv("SQUIDDLE_HOME", str(tmp_path))
    calls = []

    def fake_download(size, target, log):
        calls.append(size)
        target.mkdir(parents=True)
        (target / "inference.onnx").write_bytes(b"x")
        (target / "inference.yml").write_text("Global: {}")
        return target

    monkeypatch.setattr(models, "download_model", fake_download)
    p = models.resolve_model("small")
    assert p == tmp_path / "models" / "squiddle_PP-OCRv6_small_rec" and calls == ["small"]
    assert models.resolve_model("small") == p and calls == ["small"]      # second call hits the cache
    assert [q.name for q in models.list_cached()] == ["squiddle_PP-OCRv6_small_rec"]
    with pytest.raises(ValueError):
        models.resolve_model("large")


def test_model_repo_env_override(monkeypatch):
    monkeypatch.setenv("SQUIDDLE_MODEL_REPO", "me/squiddle-{size}")
    assert models.model_repo("tiny") == "me/squiddle-tiny"
    monkeypatch.setenv("SQUIDDLE_MODEL_REPO_TINY", "other/tiny")
    assert models.model_repo("tiny") == "other/tiny"
