import os
from pathlib import Path

import pytest
import torch

MEDIUM = Path.home() / ".local/share/htrmopo/1980fa54-b49e-5256-95b6-928290bc48f8/medium.safetensors"


@pytest.fixture(scope="session")
def tiny_net():
    """A randomly initialised PP-OCRv6 'tiny' network with a small alphabet (fast)."""
    from kraken.lib.ppocr.network import build_recognizer

    torch.manual_seed(0)
    net = build_recognizer("tiny", num_classes=12).eval()
    return net


@pytest.fixture(scope="session")
def medium_model():
    """The real kraken medium model, if it is present on this machine."""
    if not MEDIUM.is_file() or os.environ.get("SQUIDDLE_SKIP_SLOW"):
        pytest.skip("medium.safetensors not available")
    from squiddleocr.convert.loader import load_kraken_model

    return load_kraken_model(MEDIUM)


@pytest.fixture
def toy_c2l():
    # label order deliberately differs from insertion order
    return {"b": [2], " ": [1], "a": [3], "̈": [4], "ſ": [5]}
