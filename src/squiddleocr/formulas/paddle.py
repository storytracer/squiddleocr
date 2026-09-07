"""Formula recognition with PaddleX's PP-FormulaNet (``paddle`` extra).

PP-FormulaNet_plus-L and PP-FormulaNet-L ship as safetensors for PaddleX's ``transformers`` engine
(Hugging Face ``transformers`` on torch), so they run on the same torch and GPU as kraken's
recogniser; the official model source has no ONNX package for any formula model, and Paddle
Inference is not used here. A formula region's crop goes in, LaTeX comes out.
"""
from __future__ import annotations

from typing import Sequence

from ..crops import crop_bbox
from ..paddle_compat import create_transformers_predictor, paddle_image
from ..types import Page, Region

DEFAULT_FORMULA_MODEL = "PP-FormulaNet_plus-L"


class PaddleFormulaRecognizer:
    """PP-FormulaNet on PaddleX's transformers engine; crops are read in batches of ``batch_size``."""

    def __init__(self, model_name: str = DEFAULT_FORMULA_MODEL, device: str = "auto", batch_size: int = 8):
        self.model_name = model_name
        self.predictor = create_transformers_predictor(model_name, device, batch_size=batch_size)

    def recognize(self, page: Page, regions: Sequence[Region]) -> list[str]:
        crops = [paddle_image(crop_bbox(page.image, r.bbox)) for r in regions]
        todo = [i for i, c in enumerate(crops) if c.size]
        out = [""] * len(regions)
        if todo:
            results = self.predictor.predict([crops[i] for i in todo])
            for i, res in zip(todo, results):
                out[i] = str(res["rec_formula"]).strip()
        return out
