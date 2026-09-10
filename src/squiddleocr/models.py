"""Model files: kraken's by DOI (``htrmopo``, kraken's own cache) and eynollah's ONNX bundle from Zenodo.

The eynollah bundle lands in SquiddleOCR's own data directory (``$SQUIDDLE_HOME`` or
``platformdirs.user_data_dir("squiddleocr")``, ``~/.local/share/squiddleocr`` on Linux, next to
htrmopo's ``~/.local/share/htrmopo``); ``SQUIDDLE_EYNOLLAH_MODELS`` points at a bundle that already
exists. Nothing model-sized ever lives in the checkout.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

SIZES = ("tiny", "small", "medium")
DEFAULT_SIZE = "medium"
#: Zenodo DOIs of kraken's PP-OCRv6 recognisers (Benjamin Kiessling, Apache-2.0) and of the blla segmenter.
KRAKEN_DOIS = {"tiny": "10.5281/zenodo.21788403", "small": "10.5281/zenodo.21788405", "medium": "10.5281/zenodo.21788410"}
KRAKEN_SEGMENTER_DOI = "10.5281/zenodo.14602569"

Log = Callable[[str], None]


def kraken_model_file(doi: str, suffixes: tuple[str, ...] = (".safetensors", ".mlmodel"), log: Log = logger.info) -> Path:
    """The model file of a kraken model on Zenodo, via ``htrmopo.get_model`` (``~/.local/share/htrmopo``;
    downloaded on first use, served from the cache afterwards)."""
    from htrmopo import get_model

    log(f"kraken model {doi} (htrmopo cache or Zenodo)")
    folder = Path(get_model(doi))
    for suffix in suffixes:
        files = sorted(folder.glob(f"*{suffix}"))
        if files:
            return files[0]
    raise FileNotFoundError(f"{doi} was fetched to {folder} but holds no {'/'.join(suffixes)} file.")


def resolve_kraken_model(spec: str | Path = DEFAULT_SIZE, log: Log = logger.info) -> Path:
    """The recognition model for ``spec``: a size name (fetched by DOI) or a path to a kraken model file."""
    p = Path(spec)
    if p.is_file():
        return p
    size = str(spec).lower()
    if size not in SIZES:
        raise FileNotFoundError(f"{spec} is neither a model size ({', '.join(SIZES)}) nor a kraken model file.")
    return kraken_model_file(KRAKEN_DOIS[size], (".safetensors",), log)


# ---------------------------------------------------------------------------------------------- eynollah
def data_dir() -> Path:
    """SquiddleOCR's data directory: ``$SQUIDDLE_HOME``, else platformdirs' user data dir for ``squiddleocr``."""
    home = os.environ.get("SQUIDDLE_HOME")
    if home:
        return Path(home).expanduser()
    from platformdirs import user_data_dir

    return Path(user_data_dir("squiddleocr"))


@dataclass(frozen=True)
class EynollahBundle:
    """One Zenodo distribution of eynollah's inference models. Bumping to a new record is a change to
    ``EYNOLLAH_BUNDLE`` alone (and to ``files`` if the model filenames changed)."""

    version: str
    url: str
    size: int
    md5: str
    record: str
    #: model files (relative to ``models_eynollah/``, without ``.onnx``) that a layout run loads; eynollah's
    #: model zoo resolves each category to ``<name>.onnx`` (``eynollah.model_zoo.default_specs``)
    files: tuple[str, ...]

    @property
    def zip_name(self) -> str:
        return self.url.rsplit("/", 1)[-1]


#: The layout bundle of eynollah 0.9.x (``models_inference_layout_v0_9_1.zip``, Zenodo record 21381102,
#: 1.85 GB): ONNX models for column classification, page extraction, text lines, regions, full layout,
#: machine-based reading order, tables, binarisation and enhancement. Size and MD5 from the Zenodo record.
EYNOLLAH_BUNDLE = EynollahBundle(
    version="v0_9_1",
    url="https://zenodo.org/records/21381102/files/models_inference_layout_v0_9_1.zip",
    size=1847700967,
    md5="e4c23fa7deab88f736c2b7354ce28d5e",
    record="https://zenodo.org/records/21381102",
    files=(
        "eynollah-column-classifier_20210425",                 # col_classifier
        "model_eynollah_page_extraction_20250915",             # page
        "modelens_textline_0_1__2_4_16092024",                 # textline
        "modelens_e_l_all_sp_0_1_2_3_4_171024",                # region_1_2
        "modelens_full_lay_1__4_3_091124",                     # region_fl_np (-fl)
        "model_eynollah_reading_order_20250824",               # reading_order (-romb)
        "modelens_table_0t4_201124",                           # table (-tab)
    ),
)
EYNOLLAH_SUBDIR = "models_eynollah"     # eynollah's model zoo expects <basedir>/models_eynollah/<model>.onnx
EYNOLLAH_MANIFEST = "squiddle.json"


def eynollah_cache_dir(bundle: EynollahBundle = EYNOLLAH_BUNDLE) -> Path:
    """Where the bundle unpacks: ``<data dir>/eynollah/<version>/`` (eynollah's ``--model-basedir``)."""
    return data_dir() / "eynollah" / bundle.version


def eynollah_missing_files(basedir: Path, bundle: EynollahBundle = EYNOLLAH_BUNDLE) -> list[str]:
    root = Path(basedir) / EYNOLLAH_SUBDIR
    return [f for f in bundle.files if not (root / f"{f}.onnx").is_file()]


def eynollah_model_dir(bundle: EynollahBundle = EYNOLLAH_BUNDLE, download: bool = True, log: Log = logger.info,
                       progress: Callable[[int, int], None] | None = None) -> Path:
    """The directory holding ``models_eynollah/`` (eynollah's ``--model-basedir``), downloaded if needed.

    ``SQUIDDLE_EYNOLLAH_MODELS`` names a pre-existing bundle: either the basedir or the
    ``models_eynollah`` directory itself; it is never downloaded into. Otherwise the bundle lives in
    the cache; ``download=False`` raises ``FileNotFoundError`` instead of fetching it."""
    override = os.environ.get("SQUIDDLE_EYNOLLAH_MODELS")
    if override:
        d = Path(override).expanduser()
        basedir = d.parent if d.name == EYNOLLAH_SUBDIR else d
        missing = eynollah_missing_files(basedir, bundle)
        if missing:
            raise FileNotFoundError(f"SQUIDDLE_EYNOLLAH_MODELS={override}: no {EYNOLLAH_SUBDIR}/<model>.onnx for "
                                    + ", ".join(missing))
        return basedir
    basedir = eynollah_cache_dir(bundle)
    if not eynollah_missing_files(basedir, bundle):
        return basedir
    if not download:
        raise FileNotFoundError(f"eynollah models are not in {basedir}; run `squiddle models pull eynollah`")
    fetch_eynollah_models(bundle, basedir, log, progress)
    return basedir


def fetch_eynollah_models(bundle: EynollahBundle, basedir: Path, log: Log = logger.info,
                          progress: Callable[[int, int], None] | None = None) -> Path:
    """Download the bundle's zip into ``basedir`` (resumable, size- and MD5-checked), unpack it, verify the
    model files, write ``squiddle.json`` and delete the zip. Returns ``basedir``."""
    basedir = Path(basedir)
    basedir.mkdir(parents=True, exist_ok=True)
    zip_path = basedir / bundle.zip_name
    log(f"eynollah models {bundle.version} from {bundle.url} ({bundle.size / 2**30:.2f} GB) -> {basedir}")
    download_file(bundle.url, zip_path, bundle.size, bundle.md5, progress)
    log(f"unpacking {zip_path.name}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(basedir)
    _flatten_models_dir(basedir)
    missing = eynollah_missing_files(basedir, bundle)
    if missing:
        raise FileNotFoundError(f"{bundle.zip_name} unpacked into {basedir} but {EYNOLLAH_SUBDIR}/<model>.onnx is missing for "
                                + ", ".join(missing) + "; the bundle layout changed, update EYNOLLAH_BUNDLE.files")
    zip_path.unlink()
    manifest = {**asdict(bundle), "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "onnx_files": sorted(p.name for p in (basedir / EYNOLLAH_SUBDIR).rglob("*.onnx"))}
    (basedir / EYNOLLAH_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return basedir


def _flatten_models_dir(basedir: Path) -> None:
    """Put ``models_eynollah/`` directly under ``basedir`` if the zip wrapped it in another directory."""
    if (basedir / EYNOLLAH_SUBDIR).is_dir():
        return
    for candidate in sorted(basedir.glob(f"*/{EYNOLLAH_SUBDIR}")):
        shutil.move(str(candidate), str(basedir / EYNOLLAH_SUBDIR))
        try:
            candidate.parent.rmdir()
        except OSError:
            pass
        return


def download_file(url: str, dest: Path, size: int | None = None, md5: str | None = None,
                  progress: Callable[[int, int], None] | None = None, chunk: int = 1 << 20) -> Path:
    """Fetch ``url`` to ``dest`` through ``dest.part`` with HTTP range resume; verify ``size`` and ``md5``.

    A ``.part`` file from an interrupted run is continued with a ``Range`` request (a server that
    ignores it restarts the download). A size or hash mismatch removes the file and raises
    ``RuntimeError`` so the next call starts clean. ``progress(done, total)`` is called per chunk."""
    import urllib.request

    dest = Path(dest)
    if dest.is_file() and (size is None or dest.stat().st_size == size) and (md5 is None or _md5(dest) == md5):
        return dest
    part = dest.with_name(dest.name + ".part")
    done = part.stat().st_size if part.is_file() else 0
    if size is not None and done > size:
        part.unlink()
        done = 0
    if size is None or done < size:
        req = urllib.request.Request(url, headers={"User-Agent": "squiddleocr"})
        if done:
            req.add_header("Range", f"bytes={done}-")
        with urllib.request.urlopen(req, timeout=60) as resp:
            if done and resp.status != 206:      # range ignored: start over
                done = 0
            total = size if size is not None else done + int(resp.headers.get("Content-Length") or 0)
            with open(part, "ab" if done else "wb") as f:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    done += len(buf)
                    if progress:
                        progress(done, total)
    if size is not None and part.stat().st_size != size:
        got = part.stat().st_size
        part.unlink()
        raise RuntimeError(f"{url}: downloaded {got} bytes, expected {size}; removed the partial file, run again")
    if md5 is not None and (got_md5 := _md5(part)) != md5:
        part.unlink()
        raise RuntimeError(f"{url}: MD5 {got_md5} does not match {md5}; removed the file, run again")
    part.replace(dest)
    return dest


def _md5(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()
