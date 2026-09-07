import numpy as np
import pytest
import torch

from squiddleocr.convert.export import check_parity, export_onnx, make_session, run_onnx
from squiddleocr.convert.wrapper import PaddleRecWrapper, kraken_to_paddle_input

H = 96


@pytest.fixture(scope="module")
def tiny_session(tiny_net, tmp_path_factory):
    path = tmp_path_factory.mktemp("onnx") / "inference.onnx"
    export_onnx(PaddleRecWrapper(tiny_net), path, H)
    return make_session(path)


@pytest.mark.parametrize("width", [640, 1000])
def test_onnx_matches_kraken_logits(tiny_net, tiny_session, width):
    torch.manual_seed(width)
    xk = torch.rand(1, 3, H, width)
    with torch.inference_mode():
        logits, _ = tiny_net(torch.nn.functional.pad(xk, (16, 16), value=0.0), None)
    ref = logits.squeeze(2).permute(0, 2, 1).softmax(-1).numpy()
    out = run_onnx(tiny_session, kraken_to_paddle_input(xk).numpy())
    assert out.shape == ref.shape
    assert np.abs(out - ref).max() < 1e-4


def test_dynamic_batch(tiny_net, tiny_session):
    w = PaddleRecWrapper(tiny_net)
    diffs = check_parity(w, tiny_session, H, widths=(200,), batch=3)
    assert diffs[200] < 1e-4


@pytest.mark.parametrize("width", [640, 1000])
def test_real_model_export_matches_kraken(medium_model, tmp_path, width):
    net = medium_model.net
    path = tmp_path / "inference.onnx"
    wrapper = PaddleRecWrapper(net)
    export_onnx(wrapper, path, medium_model.height)
    session = make_session(path)
    torch.manual_seed(0)
    xk = torch.rand(1, 3, medium_model.height, width)
    with torch.inference_mode():
        logits, _ = net(torch.nn.functional.pad(xk, (16, 16), value=0.0), None)
    ref = logits.squeeze(2).permute(0, 2, 1).softmax(-1).numpy()
    out = run_onnx(session, kraken_to_paddle_input(xk).numpy())
    assert out.shape == ref.shape
    assert np.abs(out - ref).max() < 1e-3
