import re

import pytest
import yaml

from squiddleocr.convert.config import build_inference_config, dump_yaml, load_yaml, validate_model_name


def test_inference_config_structure():
    chars = [" ", "a", "ä", "̈", "'", "y", "null"]
    cfg = build_inference_config(chars, 96, "PP-OCRv6_medium_rec")
    assert cfg["Global"]["model_name"] == "PP-OCRv6_medium_rec"
    ops = cfg["PreProcess"]["transform_ops"]
    assert ops[0] == {"DecodeImage": {"channel_first": False, "img_mode": "RGB"}}
    assert ops[1] == {"RecResizeImg": {"image_shape": [3, 96, 96]}}
    assert cfg["PostProcess"] == {"name": "CTCLabelDecode", "character_dict": chars}
    assert cfg["Hpi"]["backend_configs"]["paddle_infer"]["trt_dynamic_shapes"]["x"][2] == [8, 3, 96, 3200]


def test_yaml_roundtrip_preserves_tricky_entries(tmp_path):
    chars = [" ", "a", "̈", "'", "y", "n", "null", "1", "#", "-", "ſ", "“"]
    cfg = build_inference_config(chars, 96, "PP-OCRv6_small_rec")
    p = tmp_path / "inference.yml"
    dump_yaml(cfg, p)
    back = load_yaml(p)
    assert back == cfg
    assert all(isinstance(c, str) for c in back["PostProcess"]["character_dict"])
    # also loadable by a plain YAML loader as PaddleX uses
    assert yaml.safe_load(p.read_text(encoding="utf-8"))["PostProcess"]["character_dict"] == chars


def test_validate_model_name():
    assert validate_model_name("PP-OCRv6_tiny_rec") == "PP-OCRv6_tiny_rec"
    with pytest.raises(ValueError):
        validate_model_name("squiddle_PP-OCRv6_medium_rec")
