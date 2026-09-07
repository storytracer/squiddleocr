"""Every 10th Fraktur test page through the SquiddleOCR pipeline (layout + PP-OCRv6 det + kraken recogniser); CER vs reference.

usage: eval_fraktur_squiddle.py [layout: paddle|none] [unclip]   -> work/eval_squiddle/<layout>/
"""
import json, shutil, sys, time, unicodedata
from pathlib import Path
sys.path.insert(0, 'src')
from squiddleocr.factory import build_pipeline
from squiddleocr.document import export
from squiddleocr.integrations.verify import edit_distance
D = Path.home() / 'data/nls/fraktur_test/images/nls.uk/aHR0cHM6Ly92aWV3Lm5scy51ay9tYW5pZmVzdC8xMzEwLzI4MTAvMTMxMDI4MTAyL21hbmlmZXN0Lmpzb24/images'
layout = sys.argv[1] if len(sys.argv) > 1 else 'paddle'
unclip = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
pages = [f'{i:04d}' for i in range(10, 231, 10)]
out = Path('work/eval_squiddle') / layout; out.mkdir(parents=True, exist_ok=True)
pipe = build_pipeline('work/squiddle_PP-OCRv6_medium_rec', layout=layout, tables=False, unclip_ratio=unclip, device='auto')
nfd = lambda t: unicodedata.normalize('NFD', t.strip())
rows = []
for page in pages:
    ref = nfd((D / f'{page}.txt').read_text(encoding='utf-8'))
    t = time.time(); doc = pipe.run_files([D / f'{page}.jpg']); dt = time.time() - t
    hyp = nfd(doc.export_to_text())
    export(doc, out, page, ['md', 'doclang'])
    (out / f'{page}.squiddle.txt').write_text(hyp, encoding='utf-8'); shutil.copy(D / f'{page}.txt', out / f'{page}.kraken.txt')
    rows.append(dict(page=page, seconds=round(dt, 2), ref_chars=len(ref), lines=len(hyp.splitlines()), ref_lines=len(ref.splitlines()),
                     cer=round(edit_distance(ref, hyp) / max(1, len(ref)), 4)))
    print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open(out / 'summary.json', 'w'), indent=1)
tot = sum(r['ref_chars'] for r in rows); reg = [r for r in rows if r['page'] not in ('0220', '0230')]
print('all 23 pages CER %.3f%% | 21 regular pages CER %.3f%% | s/page (excl. first) %.2f' % (
    100 * sum(r['cer'] * r['ref_chars'] for r in rows) / tot,
    100 * sum(r['cer'] * r['ref_chars'] for r in reg) / sum(r['ref_chars'] for r in reg),
    sum(r['seconds'] for r in rows[1:]) / (len(rows) - 1)))
