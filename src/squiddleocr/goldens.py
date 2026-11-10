"""Compare a run (page-record JSONL) with the squiddleocr 0.3 goldens (hOCR) line by line: lines present
in both (by id), line boxes equal (kraken's integer clip of the polygon), word sequences equal (text
and box). The logic of linecaster's ``tests/parity/compare_goldens.py``: the goldens' word texts are
NFC-normalised first (0.3 wrote kraken's decomposed text; the chain's ``UnicodeNormalizer`` writes NFC), so
parity means "exact after NFC"."""
from __future__ import annotations

import glob
import json
import re
import unicodedata
from pathlib import Path

BBOX = re.compile(r"bbox (\d+) (\d+) (\d+) (\d+)")


def golden(path: Path) -> dict:
    from lxml import html

    doc = html.parse(str(path))
    out = {}
    for ln in doc.xpath("//*[contains(concat(' ', normalize-space(@class), ' '), ' ocr_line ') or contains(@class,"
                        "'ocr_header') or contains(@class,'ocr_caption') or contains(@class,'ocr_textfloat')]"):
        box = tuple(int(v) for v in BBOX.search(ln.get("title")).groups())
        words = [(unicodedata.normalize("NFC", w.text_content()), tuple(int(v) for v in BBOX.search(w.get("title")).groups()))
                 for w in ln.xpath(".//*[contains(@class,'ocrx_word')]") if w.text_content().strip()]
        out[ln.get("id")] = (box, words)
    return out


def ours(row: dict, transcription: str, width: int, height: int) -> dict:
    import numpy as np

    out = {}
    for ln in row["page"]["transcriptions"][transcription]["lines"]:
        pts = np.rint(np.asarray(ln["polygon"], dtype=float)).astype(int)
        x0, y0 = int(np.clip(pts[:, 0].min(), 0, width - 1)), int(np.clip(pts[:, 1].min(), 0, height - 1))
        x1, y1 = int(np.clip(pts[:, 0].max(), 0, width - 1)), int(np.clip(pts[:, 1].max(), 0, height - 1))
        words = [(w["text"], (int(w["bbox"][0]), int(w["bbox"][1]), int(round(w["bbox"][0] + w["bbox"][2])),
                              int(round(w["bbox"][1] + w["bbox"][3])))) for w in ln["words"]]
        out[ln["id"]] = ((x0, y0, x1, y1), words)
    return out


def compare(run: str | Path, goldens: str | Path, transcription: str = "ocr_new", verbose: bool = False) -> dict:
    rows = [json.loads(line) for f in sorted(glob.glob(f"{run}/**/*.jsonl", recursive=True)) for line in open(f)]
    tot = dict(pages=0, lines_gold=0, lines_ours=0, common=0, box=0, words=0, pages_exact=0)
    for row in sorted(rows, key=lambda r: r["id"]):
        g = golden(next(Path(goldens).glob(f"{row['id']}.*hocr")))
        img = row["page"]["image"]
        o = ours(row, transcription, img["width"], img["height"])
        common = set(g) & set(o)
        box = sum(g[k][0] == o[k][0] for k in common)
        words = sum(g[k][1] == o[k][1] for k in common)
        exact = len(g) == len(o) == len(common) == box == words
        for k, v in (("pages", 1), ("lines_gold", len(g)), ("lines_ours", len(o)), ("common", len(common)),
                     ("box", box), ("words", words), ("pages_exact", int(exact))):
            tot[k] += v
        if not exact and verbose:
            print(f"{row['id']}: gold {len(g)} ours {len(o)} common {len(common)} box {box} words {words}")
    return tot
