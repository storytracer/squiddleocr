"""Generate PaddleX ``inference.yml`` for a SquiddleOCR recognition model directory."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from . import PADDLE_MODEL_NAMES

#: Maximum width PaddleX's OCRReisizeNormImg will feed at a given height
#: (``max_imgW = 3200`` is hardcoded in PaddleX; wider lines are squashed).
PADDLE_MAX_WIDTH = 3200


def build_inference_config(
    chars: list[str],
    height: int,
    model_name: str,
    min_width: int | None = None,
) -> dict[str, Any]:
    """Build the ``inference.yml`` structure PaddleX 3.x expects for a CTC recogniser.

    ``model_name`` must be a name PaddleX registers for the text recognition
    runner (``PP-OCRv6_<size>_rec``); PaddleX rejects
    unknown names and names that differ from the requested model name.

    ``min_width`` is the width floor PaddleX pads lines to (``RecResizeImg.image_shape``).
    It defaults to ``height``, i.e. effectively no padding, because kraken does not
    pad lines to a minimum width. Stock PaddleOCR models use 320.
    """
    if min_width is None:
        min_width = height
    shapes = [[1, 3, height, min_width], [1, 3, height, 2 * min_width], [8, 3, height, PADDLE_MAX_WIDTH]]
    return {
        "Global": {"model_name": model_name},
        "Hpi": {
            "backend_configs": {
                "paddle_infer": {"trt_dynamic_shapes": {"x": shapes}},
                "tensorrt": {"dynamic_shapes": {"x": shapes}},
            }
        },
        "PreProcess": {
            "transform_ops": [
                # kraken lines are RGB; PaddleX's default is BGR.
                {"DecodeImage": {"channel_first": False, "img_mode": "RGB"}},
                # Height is the only preprocessing knob PaddleX exposes. The
                # 0..1 scaling, inversion and white padding kraken applies are
                # folded into the ONNX graph (see wrapper.py).
                {"RecResizeImg": {"image_shape": [3, height, min_width]}},
                {"KeepKeys": {"keep_keys": ["image"]}},
            ]
        },
        "PostProcess": {
            "name": "CTCLabelDecode",
            # Entry i is CTC class i+1. PaddleX prepends "blank" and, because
            # use_space_char defaults to True, appends one more " " at the end;
            # that trailing index is never emitted by the network.
            "character_dict": list(chars),
        },
    }


def dump_yaml(config: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False, default_flow_style=False, width=1000),
        encoding="utf-8",
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def validate_model_name(model_name: str) -> str:
    known = set(PADDLE_MODEL_NAMES.values())
    if model_name not in known:
        raise ValueError(
            f"{model_name!r} is not a PaddleX text recognition name this tool knows to work: {sorted(known)}"
        )
    return model_name
