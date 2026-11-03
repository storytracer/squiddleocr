---
schema: linecaster.recipe/1
id: squiddle
edition: 1
pipeline: chain/1
presets:
  layout: pp-doclayoutv3@1
  lines: pp-ocrv6-medium-det@1
  recognition: kraken-ppocrv6-medium@1
  tables: slanet-plus@2
  formulas: pp-formulanet-plus-l@1
options:
  text: reflow
provenance:
  derived_from:
  - storytracer/squiddleocr 0.3, the paddle pipeline's defaults (`squiddle legacy ocr --pipeline paddle`)
  - 'storytracer/squiddleocr@7c8a052 (the goldens: sample-60, tables-7)'
  authored_by:
  - Sebastian Majstorovic
  created: 2026-11-03
support: core
---

SquiddleOCR's combination: fast OCR for historical print from small models. PP-DocLayoutV3 finds the regions
and their reading order, PP-OCRv6 medium detection finds the lines on the whole page, kraken's PP-OCRv6 medium
recogniser reads them (batch 8, bf16 on CUDA), SLANet_plus rebuilds tables and PP-FormulaNet_plus-L reads
formulas as LaTeX; the text is reflowed into paragraphs.

- The settings squiddleocr 0.3 shipped and its goldens were recorded with; the chain reproduces those goldens
  line for line (see the evidence).
- `text: reflow` is `squiddle ocr`'s default; the goldens compare lines, which the text mode does not change.
- `squiddle ocr` runs this recipe when its model options are left at their defaults; with other models it is a
  variation, not the recipe.
