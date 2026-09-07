"""PP-StructureV3 + SquiddleOCR recogniser on every 10th Fraktur test page; CER vs reference."""
import json, shutil, sys, time, unicodedata
from pathlib import Path
sys.path.insert(0, 'src')
from squiddleocr.verify import edit_distance
D = Path.home() / 'data/nls/fraktur_test/images/nls.uk/aHR0cHM6Ly92aWV3Lm5scy51ay9tYW5pZmVzdC8xMzEwLzI4MTAvMTMxMDI4MTAyL21hbmlmZXN0Lmpzb24/images'
K = Path.home() / 'data/nls/fraktur_test/output/kraken/txt'
pages = [f'{i:04d}' for i in range(10, 231, 10)]
unclip = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
from paddleocr import PPStructureV3
pipe = PPStructureV3(paddlex_config='work/PP-StructureV3_squiddle.yaml', device='gpu', engine='onnxruntime',
                     use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
                     use_formula_recognition=False, use_chart_recognition=False, use_seal_recognition=False)
out = Path('work/eval_every10'); out.mkdir(exist_ok=True)
nfd = lambda t: unicodedata.normalize('NFD', t.strip())
rows = []
for page in pages:
    ref = nfd((D / f'{page}.txt').read_text(encoding='utf-8'))
    t = time.time()
    texts = []
    for res in pipe.predict(str(D / f'{page}.jpg'), text_det_unclip_ratio=unclip):
        res.save_to_markdown(save_path=str(out / page))
        texts += res.json['res'].get('overall_ocr_res', {}).get('rec_texts', [])
    dt = time.time() - t
    hyp = nfd('\n'.join(texts))
    # flat layout: NNNN.squiddle.txt / NNNN.squiddle.md / NNNN.kraken.txt (the reference)
    (out / f'{page}.squiddle.txt').write_text(hyp, encoding='utf-8')
    md = out / page / f'{page}.md'
    if md.exists():
        md.replace(out / f'{page}.squiddle.md')
        shutil.rmtree(out / page, ignore_errors=True)
    shutil.copy(D / f'{page}.txt', out / f'{page}.kraken.txt')
    kr = nfd((K / f'{page}.txt').read_text(encoding='utf-8')) if (K / f'{page}.txt').exists() else None
    row = dict(page=page, seconds=round(dt, 2), ref_lines=len(ref.splitlines()), lines=len(texts), ref_chars=len(ref),
               cer=round(edit_distance(ref, hyp) / max(1, len(ref)), 4),
               kraken_cer=round(edit_distance(ref, kr) / max(1, len(ref)), 4) if kr is not None else None)
    print(json.dumps(row), flush=True)
    rows.append(row)
json.dump(rows, open(out / 'summary.json', 'w'), indent=1)
tot = sum(r['ref_chars'] for r in rows)
print('pages', len(rows), 'total ref chars', tot)
print('squiddle CER (char-weighted): %.3f%%' % (100 * sum(r['cer'] * r['ref_chars'] for r in rows) / tot))
kr = [r for r in rows if r['kraken_cer'] is not None]
if kr: print('kraken   CER (char-weighted): %.3f%%' % (100 * sum(r['kraken_cer'] * r['ref_chars'] for r in kr) / sum(r['ref_chars'] for r in kr)))
print('seconds/page (excl. first): %.2f' % (sum(r['seconds'] for r in rows[1:]) / max(1, len(rows) - 1)))
