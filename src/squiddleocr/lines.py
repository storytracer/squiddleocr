"""Cut text-line images out of page images with kraken's segmenter (for `squiddle verify`)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

HTRMOPO_BLLA = Path.home() / ".local/share/htrmopo/97665cf3-f83d-5594-8855-f28d3af9df7a/blla.mlmodel"


def default_seg_model() -> Path:
    if HTRMOPO_BLLA.is_file():
        return HTRMOPO_BLLA
    raise FileNotFoundError(
        "No kraken segmentation model found; pass --seg-model (e.g. kraken get 10.5281/zenodo.10592716)."
    )


def extract_lines(
    pages: Iterable[Path],
    out_dir: Path,
    seg_model: Path | None = None,
    max_lines: int = 0,
    device: str = "cpu",
    echo: Callable[[str], None] = print,
) -> int:
    """Segment pages with kraken and write ``<page>_<nnnn>.png`` per line plus ``lines.json``.

    The line images are exactly what kraken's recogniser would see before its
    line transform (polygon-masked strips with white background), so `verify`
    compares recognisers, not segmenters.
    """
    from kraken.configs import SegmentationInferenceConfig
    from kraken.lib.segmentation import extract_polygons
    from kraken.lib.util import open_image
    from kraken.tasks.segmentation import SegmentationTaskModel

    seg_model = seg_model or default_seg_model()
    model = SegmentationTaskModel.load_model(str(seg_model))
    cfg = SegmentationInferenceConfig(device=device if device != "cpu" else "auto",
                                      accelerator="cpu" if device == "cpu" else "auto")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    n = 0
    for page in pages:
        page = Path(page)
        im = open_image(str(page))
        echo(f"Segmenting {page.name}")
        seg = model.predict(im=im, config=cfg)
        for i, (line_im, line) in enumerate(extract_polygons(im, seg)):
            if line_im is None or 0 in line_im.size:
                continue
            name = f"{page.stem}_{i:04d}.png"
            line_im.convert("RGB").save(out_dir / name)
            index.append({"file": name, "page": page.name, "index": i,
                          "baseline": getattr(line, "baseline", None),
                          "boundary": getattr(line, "boundary", None)})
            n += 1
            if max_lines and n >= max_lines:
                break
        if max_lines and n >= max_lines:
            break
    (out_dir / "lines.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return n
