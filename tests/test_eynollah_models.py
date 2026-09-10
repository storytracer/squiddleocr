"""The eynollah model manager against a local HTTP server with range support (nothing from Zenodo)."""
import hashlib
import io
import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from squiddleocr import models
from squiddleocr.models import (EYNOLLAH_SUBDIR, EynollahBundle, download_file, eynollah_model_dir, fetch_eynollah_models)


def make_zip(files, wrapper=""):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in files:
            zf.writestr(f"{wrapper}{EYNOLLAH_SUBDIR}/{name}.onnx", b"onnx" * 500 + name.encode())
        zf.writestr(f"{wrapper}{EYNOLLAH_SUBDIR}/extra.onnx", b"x" * 100)
    return buf.getvalue()


class Server:
    """Serves one payload at /bundle.zip with Range support and records the requests it saw."""

    def __init__(self, payload: bytes, ignore_range: bool = False):
        self.payload, self.ignore_range, self.requests = payload, ignore_range, []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                rng = self.headers.get("Range")
                server.requests.append(rng)
                data = server.payload
                if rng and not server.ignore_range:
                    start = int(rng.split("=")[1].rstrip("-"))
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                    data = data[start:]
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/bundle.zip"

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def bundle_files():
    return ("model_a", "model_b")


@pytest.fixture
def served(bundle_files):
    payload = make_zip(bundle_files)
    s = Server(payload)
    yield s, payload
    s.close()


def _bundle(server, payload, files):
    return EynollahBundle(version="vtest", url=server.url, size=len(payload), md5=hashlib.md5(payload).hexdigest(),
                          record="http://example.invalid/record", files=files)


def test_download_resume_and_hash(tmp_path, served):
    server, payload = served
    dest = tmp_path / "bundle.zip"
    md5 = hashlib.md5(payload).hexdigest()
    (tmp_path / "bundle.zip.part").write_bytes(payload[:1000])          # an interrupted earlier run
    progress = []
    download_file(server.url, dest, len(payload), md5, progress=lambda d, t: progress.append((d, t)))
    assert dest.read_bytes() == payload and not (tmp_path / "bundle.zip.part").exists()
    assert server.requests == ["bytes=1000-"] and progress[-1] == (len(payload), len(payload))
    download_file(server.url, dest, len(payload), md5)                   # complete file: no request
    assert server.requests == ["bytes=1000-"]


def test_download_restarts_when_range_is_ignored(tmp_path):
    payload = make_zip(("m",))
    server = Server(payload, ignore_range=True)
    try:
        (tmp_path / "b.zip.part").write_bytes(payload[:500])
        download_file(server.url, tmp_path / "b.zip", len(payload), hashlib.md5(payload).hexdigest())
        assert (tmp_path / "b.zip").read_bytes() == payload
    finally:
        server.close()


def test_download_rejects_a_wrong_hash_and_size(tmp_path, served):
    server, payload = served
    with pytest.raises(RuntimeError, match="MD5"):
        download_file(server.url, tmp_path / "b.zip", len(payload), "0" * 32)
    assert not list(tmp_path.iterdir())
    with pytest.raises(RuntimeError, match="expected"):
        download_file(server.url, tmp_path / "b.zip", len(payload) + 1, None)
    assert not list(tmp_path.iterdir())


def test_fetch_unpacks_verifies_and_writes_manifest(tmp_path, served, bundle_files):
    server, payload = served
    bundle = _bundle(server, payload, bundle_files)
    log = []
    basedir = fetch_eynollah_models(bundle, tmp_path / "v", log=log.append)
    assert basedir == tmp_path / "v"
    assert sorted(p.name for p in (basedir / EYNOLLAH_SUBDIR).iterdir()) == ["extra.onnx", "model_a.onnx", "model_b.onnx"]
    assert not (basedir / "bundle.zip").exists()
    manifest = json.loads((basedir / "squiddle.json").read_text())
    assert manifest["url"] == server.url and manifest["md5"] == bundle.md5 and manifest["version"] == "vtest"
    assert manifest["onnx_files"] == ["extra.onnx", "model_a.onnx", "model_b.onnx"] and "date" in manifest
    assert any("unpacking" in line for line in log)


def test_fetch_flattens_a_wrapper_directory_and_reports_missing_models(tmp_path):
    payload = make_zip(("model_a",), wrapper="models_layout_vtest/")
    server = Server(payload)
    try:
        ok = _bundle(server, payload, ("model_a",))
        basedir = fetch_eynollah_models(ok, tmp_path / "ok")
        assert (basedir / EYNOLLAH_SUBDIR / "model_a.onnx").is_file() and not (basedir / "models_layout_vtest").exists()
        bad = _bundle(server, payload, ("model_a", "model_zzz"))
        with pytest.raises(FileNotFoundError, match="model_zzz"):
            fetch_eynollah_models(bad, tmp_path / "bad")
    finally:
        server.close()


def test_model_dir_uses_cache_env_and_override(tmp_path, monkeypatch, served, bundle_files):
    server, payload = served
    bundle = _bundle(server, payload, bundle_files)
    monkeypatch.setenv("SQUIDDLE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SQUIDDLE_EYNOLLAH_MODELS", raising=False)
    assert models.data_dir() == tmp_path / "home"
    with pytest.raises(FileNotFoundError, match="models pull"):
        eynollah_model_dir(bundle, download=False)
    basedir = eynollah_model_dir(bundle)
    assert basedir == tmp_path / "home" / "eynollah" / "vtest" and (basedir / EYNOLLAH_SUBDIR / "model_a.onnx").is_file()
    n = len(server.requests)
    assert eynollah_model_dir(bundle) == basedir and len(server.requests) == n     # second call: no download

    monkeypatch.setenv("SQUIDDLE_EYNOLLAH_MODELS", str(basedir / EYNOLLAH_SUBDIR))
    assert eynollah_model_dir(bundle, download=False) == basedir                   # the models_eynollah dir itself
    monkeypatch.setenv("SQUIDDLE_EYNOLLAH_MODELS", str(basedir))
    assert eynollah_model_dir(bundle, download=False) == basedir
    monkeypatch.setenv("SQUIDDLE_EYNOLLAH_MODELS", str(tmp_path / "nowhere"))
    with pytest.raises(FileNotFoundError, match="SQUIDDLE_EYNOLLAH_MODELS"):
        eynollah_model_dir(bundle)
