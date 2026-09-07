import pytest

from squiddleocr import models


def _fake_model_dir(d):
    d.mkdir(parents=True)
    (d / "inference.onnx").write_bytes(b"x")
    (d / "inference.yml").write_text("Global: {}")
    return d


def test_resolve_path_passthrough(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        models.resolve_model(d)
    _fake_model_dir(tmp_path / "ok")
    assert models.resolve_model(tmp_path / "ok") == tmp_path / "ok"


def test_resolve_from_local_source_folder(tmp_path):
    src = tmp_path / "squiddleocr-models"
    small = _fake_model_dir(src / "models" / "squiddle_PP-OCRv6_small_rec")
    assert models.resolve_model("small", src) == small
    assert models.list_models(src) == [("small", small)]
    with pytest.raises(FileNotFoundError):
        models.resolve_model("tiny", src)


def test_resolve_repo_uses_cache_then_download(tmp_path, monkeypatch):
    monkeypatch.setenv("SQUIDDLE_HOME", str(tmp_path))
    calls = []

    def fake_download(repo, size, target, log):
        calls.append((repo, size))
        return _fake_model_dir(target / "models" / f"squiddle_PP-OCRv6_{size}_rec")

    monkeypatch.setattr(models, "download_model", fake_download)
    p = models.resolve_model("small", "someone/repo")
    assert p == tmp_path / "someone--repo" / "models" / "squiddle_PP-OCRv6_small_rec" and calls == [("someone/repo", "small")]
    assert models.resolve_model("small", "someone/repo") == p and len(calls) == 1      # cache hit
    assert models.list_models("someone/repo") == [("small", p)]
    with pytest.raises(ValueError):
        models.resolve_model("large")


def test_default_source_env(monkeypatch):
    assert models.default_source() == "storytracer/squiddleocr"
    monkeypatch.setenv("SQUIDDLE_MODELS", "me/mine")
    assert models.default_source() == "me/mine"
