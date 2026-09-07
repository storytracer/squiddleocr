"""Model card, licence and provenance files for a converted model directory."""
from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any

MODEL_CARD_HEADER = """\
<!--
This model card is reproduced unchanged from the kraken model repository entry
of the source model. The weights in this directory are a format conversion of
that model performed by SquiddleOCR (https://github.com/storytracer/squiddleocr);
the "squiddle_" directory name is SquiddleOCR's naming and does not imply any
endorsement by the model author. Please cite the DOI below when using the model.
-->

"""


def find_model_card(source: Path) -> Path | None:
    """kraken/htrmopo stores the model card as README.md next to the weights."""
    for name in ("README.md", "readme.md", "README.txt"):
        p = source.parent / name
        if p.is_file():
            return p
    return None


def parse_front_matter(text: str) -> dict[str, Any]:
    """Parse the YAML front matter of an htrmopo model card, if any."""
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return {}
    import yaml

    try:
        data = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def card_summary(card: Path | None) -> dict[str, Any]:
    if card is None:
        return {}
    fm = parse_front_matter(card.read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    if "id" in fm:
        out["doi"] = str(fm["id"])
    if "authors" in fm:
        out["authors"] = fm["authors"]
    for k in ("summary", "license", "software_name"):
        if k in fm:
            out[k] = fm[k]
    return out


def write_model_card(card: Path | None, out_dir: Path) -> Path | None:
    if card is None:
        return None
    dst = out_dir / "MODEL_CARD.md"
    dst.write_text(MODEL_CARD_HEADER + card.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


def write_provenance(out_dir: Path, info: dict[str, Any]) -> Path:
    info = dict(info)
    info.setdefault("created", _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    dst = out_dir / "squiddle.json"
    dst.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dst


def write_notice(out_dir: Path, summary: dict[str, Any], variant: str) -> Path:
    authors = summary.get("authors") or []
    names = ", ".join(
        f"{a.get('name')} ({a.get('affiliation')})" if isinstance(a, dict) and a.get("affiliation") else str(a.get("name") if isinstance(a, dict) else a)
        for a in authors
    ) or "the original author"
    doi = summary.get("doi", "unknown")
    lic = summary.get("license", "Apache-2.0")
    text = f"""SquiddleOCR model directory: squiddle_PP-OCRv6_{variant}_rec

Source model: kraken PP-OCRv6 ({variant}) text recognition model
Author(s):    {names}
Licence:      {lic}
DOI:          {doi}

The weights here are a format conversion (PyTorch safetensors -> ONNX with the
kraken preprocessing folded into the graph) of the model above. Nothing was
retrained. The "squiddle_" prefix identifies the packager, SquiddleOCR, and is
not part of the author's naming. See MODEL_CARD.md for the original model card.
"""
    dst = out_dir / "NOTICE"
    dst.write_text(text, encoding="utf-8")
    return dst
