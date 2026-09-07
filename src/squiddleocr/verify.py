"""Compare kraken's recogniser with the exported model on real line images."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .codec import ctc_greedy_decode, read_dict_file
from .config import load_yaml
from .export import make_session, run_onnx
from .wrapper import DEFAULT_PADDING, kraken_to_paddle_input

HTRMOPO = Path.home() / ".local/share/htrmopo"


# ----------------------------------------------------------------- utilities
def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass
class Comparison:
    name: str
    texts: list[str]
    exact: int = 0
    edits: int = 0
    ref_chars: int = 0
    mismatches: list[tuple[str, str, str]] = field(default_factory=list)  # file, ref, hyp
    extra: dict = field(default_factory=dict)

    def score(self, files: list[Path], ref: list[str]) -> None:
        for f, r, h in zip(files, ref, self.texts):
            self.ref_chars += len(r)
            d = edit_distance(r, h)
            self.edits += d
            if d == 0:
                self.exact += 1
            else:
                self.mismatches.append((f.name, r, h))

    @property
    def cer(self) -> float:
        return self.edits / self.ref_chars if self.ref_chars else math.nan

    def summary(self, n: int) -> dict:
        return {"name": self.name, "exact": self.exact, "total": n, "exact_rate": self.exact / n if n else math.nan,
                "cer_vs_kraken": self.cer, **self.extra}


def find_kraken_model(model_dir: Path) -> Path:
    """Locate the source kraken model via squiddle.json (same directory or htrmopo cache)."""
    info = json.loads((model_dir / "squiddle.json").read_text(encoding="utf-8"))
    name = info.get("source_file")
    sha = info.get("source_sha256")
    candidates = [model_dir / name] if name else []
    if HTRMOPO.is_dir():
        candidates += sorted(HTRMOPO.glob(f"*/{name}")) if name else []
        candidates += sorted(HTRMOPO.glob("*/*.safetensors"))
    for c in candidates:
        if c.is_file():
            if sha:
                import hashlib

                if hashlib.sha256(c.read_bytes()).hexdigest() != sha:
                    continue
            return c
    raise FileNotFoundError("Source kraken model not found; pass --kraken-model.")


# ------------------------------------------------------------ kraken side
def kraken_transform(height: int, padding: int, dtype=torch.float32):
    """kraken's line transform: RGB, resize to ``height``, pad white, 0..1, invert."""
    from kraken.lib.dataset.utils import ImageInputTransforms

    return ImageInputTransforms(1, height, 0, 3, (padding, 0), False, dtype=dtype)


def kraken_reference(km, files: list[Path], device: str = "cpu", padding: int = DEFAULT_PADDING):
    """Run kraken's network on each line exactly as kraken's inference path does.

    Returns decoded texts, per-line softmax outputs ``(W', C)`` and the unpadded
    kraken tensors (for feeding the ONNX model in "kraken space").
    """
    from kraken.lib.ctc_decoder import greedy_decoder

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    net = km.net.eval().to(device)
    tf_pad = kraken_transform(km.height, padding)
    tf_nopad = kraken_transform(km.height, 0)
    texts, probs, tensors = [], [], []
    with torch.inference_mode():
        for f in files:
            im = Image.open(f)
            x = tf_pad(im).unsqueeze(0).to(device)
            logits, _ = net(x, None)
            p = logits.softmax(1).squeeze(2)[0].cpu()        # (C, W')
            dec = km.model.codec.decode(greedy_decoder(p.unsqueeze(0), None)[0])
            texts.append("".join(c for c, *_ in dec))
            probs.append(p.T.numpy())
            tensors.append(tf_nopad(im))
    return texts, probs, tensors


def kraken_batched(km, tensors: list[torch.Tensor], batch_size: int, padding: int, device: str) -> list[str]:
    """kraken's batch mode: pad lines to the batch width with white and mask by length."""
    from kraken.lib.ctc_decoder import greedy_decoder

    net = km.net.eval().to(device)
    out = []
    with torch.inference_mode():
        for i in range(0, len(tensors), batch_size):
            chunk = [F.pad(t, (padding, padding), value=0.0) for t in tensors[i:i + batch_size]]
            w = max(t.shape[2] for t in chunk)
            x = torch.stack([F.pad(t, (0, w - t.shape[2])) for t in chunk]).to(device)
            lens = torch.tensor([t.shape[2] for t in chunk])
            logits, olens = net(x, lens)
            probs = logits.softmax(1).squeeze(2).cpu()
            for p, ol in zip(probs, olens):
                dec = km.model.codec.decode(greedy_decoder(p[..., :ol].unsqueeze(0), None)[0])
                out.append("".join(c for c, *_ in dec))
    return out


# ---------------------------------------------------------- paddle-style side
def paddle_preprocess(im: Image.Image, height: int, min_width: int, max_width: int = 3200) -> np.ndarray:
    """Replicate PaddleX ``OCRReisizeNormImg.resize`` for one image.

    RGB, cv2 bilinear resize to ``height``, ``[-1, 1]`` normalisation, right
    padding with 0 up to ``min_width`` (``RecResizeImg.image_shape`` width) and
    squashing of lines wider than ``max_width`` (hardcoded 3200 in PaddleX).
    """
    import cv2

    arr = np.asarray(im.convert("RGB"))
    h, w = arr.shape[:2]
    ratio = w / float(h)
    img_w = int(height * max(min_width / float(height), ratio))
    if img_w > max_width:
        resized_w = img_w = max_width
    else:
        resized_w = min(int(math.ceil(height * ratio)), img_w)
    resized = cv2.resize(arr, (resized_w, height))       # INTER_LINEAR, as PaddleX
    x = resized.astype("float32").transpose(2, 0, 1) / 255
    x = (x - 0.5) / 0.5
    out = np.zeros((3, height, img_w), dtype=np.float32)
    out[:, :, :resized_w] = x
    return out


def run_onnx_batched(session, xs: list[np.ndarray], batch_size: int) -> list[np.ndarray]:
    """Batch like PaddleX ToBatch: right-pad with 0 (grey in [-1,1] space) to the widest in the batch."""
    outs = []
    for i in range(0, len(xs), batch_size):
        chunk = xs[i:i + batch_size]
        w = max(x.shape[2] for x in chunk)
        batch = np.zeros((len(chunk), 3, chunk[0].shape[1], w), dtype=np.float32)
        for j, x in enumerate(chunk):
            batch[j, :, :, : x.shape[2]] = x
        y = run_onnx(session, batch)
        outs.extend(y[j] for j in range(len(chunk)))
    return outs


def paddlex_predict(model_dir: Path, model_name: str, files: list[Path], batch_size: int) -> list[str]:
    from paddlex.inference import create_predictor

    pred = create_predictor(model_name, model_dir=str(model_dir), engine="onnxruntime", device="cpu",
                            batch_size=batch_size)
    return [r["rec_text"] for r in pred.predict([str(f) for f in files], batch_size=batch_size)]


# ------------------------------------------------------------------ driver
def run_verify(model_dir: Path, files: list[Path], kraken_model: Path | None = None, batch_size: int = 8,
               use_paddle: bool = True, device: str = "cpu", report: Path | None = None, show: int = 20,
               echo: Callable[[str], None] = print) -> bool:
    from .loader import load_kraken_model

    model_dir = Path(model_dir)
    cfg = load_yaml(model_dir / "inference.yml")
    model_name = cfg["Global"]["model_name"]
    shape = next(op["RecResizeImg"]["image_shape"] for op in cfg["PreProcess"]["transform_ops"] if "RecResizeImg" in op)
    height, min_width = int(shape[1]), int(shape[2])
    chars_yml = cfg["PostProcess"]["character_dict"]
    chars_txt = read_dict_file(model_dir / "dict.txt")
    if chars_yml != chars_txt:
        raise RuntimeError("dict.txt and inference.yml character_dict differ")
    info = json.loads((model_dir / "squiddle.json").read_text(encoding="utf-8"))
    padding = int(info.get("input", {}).get("padding_px", DEFAULT_PADDING))

    kraken_model = Path(kraken_model) if kraken_model else find_kraken_model(model_dir)
    echo(f"kraken model: {kraken_model}")
    km = load_kraken_model(kraken_model)
    if km.height != height:
        raise RuntimeError(f"inference.yml height {height} != kraken model height {km.height}")
    chars_kraken = [c for c, _ in sorted(km.c2l.items(), key=lambda kv: kv[1][0])]
    if chars_kraken != chars_txt:
        raise RuntimeError("dictionary does not match the kraken codec")

    n = len(files)
    echo(f"{n} line images; kraken reference on {device}")
    ref, ref_probs, tensors = kraken_reference(km, files, device=device, padding=padding)
    results: list[Comparison] = []

    session = make_session(model_dir / "inference.onnx")

    # A. ONNX on kraken's own tensor: proves the export + folded preprocessing.
    texts, maxdiff = [], 0.0
    for t, p_ref in zip(tensors, ref_probs):
        x = kraken_to_paddle_input(t).unsqueeze(0).numpy()
        p = run_onnx(session, x)[0]
        if p.shape == p_ref.shape:
            maxdiff = max(maxdiff, float(np.abs(p - p_ref).max()))
        else:
            maxdiff = math.inf
        texts.append(ctc_greedy_decode(p, chars_txt)[0])
    a = Comparison("onnx_kraken_tensor", texts, extra={"max_abs_prob_diff": maxdiff}); a.score(files, ref); results.append(a)

    # B. ONNX with PaddleX-style preprocessing (cv2 bilinear resize), batch 1.
    xs = [paddle_preprocess(Image.open(f), height, min_width) for f in files]
    b = Comparison("onnx_paddle_preproc_batch1", [ctc_greedy_decode(p, chars_txt)[0] for p in run_onnx_batched(session, xs, 1)])
    b.score(files, ref); results.append(b)

    # C. Same, batched with grey right-padding like PaddleX ToBatch.
    if batch_size > 1:
        c = Comparison(f"onnx_paddle_preproc_batch{batch_size}",
                       [ctc_greedy_decode(p, chars_txt)[0] for p in run_onnx_batched(session, xs, batch_size)])
        c.score(files, ref); results.append(c)

    # C'. kraken's own batched inference (white padding + attention mask, as `kraken ocr -b N`),
    #     to show how much of the batch effect is inherent to the model.
    if batch_size > 1:
        texts = kraken_batched(km, tensors, batch_size, padding, device)
        k = Comparison(f"kraken_native_batch{batch_size}", texts); k.score(files, ref); results.append(k)

    # D. The real PaddleX predictor, if available.
    if use_paddle:
        try:
            import paddlex  # noqa: F401
        except ImportError:
            echo("paddlex not importable; skipping the PaddleX predictor run")
        else:
            for bs in sorted({1, batch_size}):
                d = Comparison(f"paddlex_predictor_batch{bs}", paddlex_predict(model_dir, model_name, files, bs))
                d.score(files, ref); results.append(d)

    echo("")
    echo(f"{'comparison':36s} {'exact':>7s} {'rate':>7s} {'CER':>8s}")
    for r in results:
        s = r.summary(n)
        line = f"{r.name:36s} {r.exact:7d} {s['exact_rate']:7.3%} {r.cer:8.4%}"
        if "max_abs_prob_diff" in r.extra:
            line += f"   max|Δp|={r.extra['max_abs_prob_diff']:.2e}"
        echo(line)
    for r in results:
        if r.mismatches:
            echo(f"\n{r.name}: {len(r.mismatches)} mismatching lines (showing {min(show, len(r.mismatches))})")
            for f, rr, h in r.mismatches[:show]:
                echo(f"  {f}\n    kraken: {rr}\n    other : {h}")
    if report:
        Path(report).write_text(json.dumps({
            "model_dir": str(model_dir), "kraken_model": str(kraken_model), "lines": n,
            "results": [{**r.summary(n), "mismatches": [{"file": f, "kraken": rr, "other": h} for f, rr, h in r.mismatches]}
                        for r in results],
        }, indent=1, ensure_ascii=False), encoding="utf-8")
        echo(f"\nreport written to {report}")
    return results[0].exact == n
