"""Where the stages get their eynollah pages from: a directory of PAGE-XML, filled by the runner.

One ``EynollahSource`` is shared by ``EynollahLayout`` and ``EynollahLines``; it parses each page's
XML once. ``prepare(paths)`` runs eynollah on a whole batch up front (its page jobs run in parallel);
a page that was not prepared is run on its own when a stage first asks for it. With an
``xml_dir`` the XML files stay there and pages that already have one are never re-run; without one
a temporary directory is used and removed on ``close``.
"""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from ..types import Page
from .pagexml import EynollahPage, parse_page_xml
from .resources import Policy, parse_vram_margin
from .runner import EynollahRunner, RunResult

DEFAULT_ARGS = "-fl -romb"


@dataclass
class EynollahOptions:
    """The CLI's eynollah settings."""

    jobs: int | None = None
    device: str | None = None
    vram_margin: str | None = None
    tensorrt: bool = False
    args: str = DEFAULT_ARGS
    xml_dir: Path | None = None
    model_dir: Path | None = None
    rtl: bool = False
    baselines: bool = True
    descender: float = 0.18
    lines: str = "paddle"            # line source: "paddle" (PP-OCRv6 det per region), "eynollah" (its polygons), "blla"
    crop_pad: int = 20               # pixels around a region's box for the per-region detector crop
    det_model: str = "PP-OCRv6_medium_det"
    unclip_ratio: float = 2.0
    page_numbers: bool = True        # page numbers from PP-DocLayoutV3's number class (needs the paddle extra)

    def policy(self) -> Policy:
        import shlex

        args = tuple(shlex.split(self.args or ""))
        if self.rtl and not any(a in ("-r2l", "--right2left") for a in args):
            args += ("-r2l",)
        return Policy(jobs=self.jobs, device=self.device, vram_margin=parse_vram_margin(self.vram_margin),
                      tensorrt=self.tensorrt, args=args)


@dataclass
class EynollahSource:
    options: EynollahOptions = field(default_factory=EynollahOptions)
    log: Callable[[str], None] = lambda s: None
    warn: Callable[[str], None] | None = None
    progress: Callable[[int, int], None] | None = None
    _runner: EynollahRunner | None = None
    _xml_dir: Path | None = None
    _temp: Path | None = None
    _pages: dict[Path, EynollahPage] = field(default_factory=dict)
    last_result: RunResult | None = None

    @property
    def xml_dir(self) -> Path:
        if self._xml_dir is None:
            if self.options.xml_dir is not None:
                self._xml_dir = Path(self.options.xml_dir)
                self._xml_dir.mkdir(parents=True, exist_ok=True)
            else:
                self._temp = self._xml_dir = Path(tempfile.mkdtemp(prefix="squiddle-eynollah-"))
        return self._xml_dir

    @property
    def runner(self) -> EynollahRunner:
        if self._runner is None:
            import importlib.util

            from ..models import eynollah_model_dir

            if importlib.util.find_spec("eynollah") is None:
                raise RuntimeError('eynollah is not installed: pip install "squiddleocr[eynollah]" '
                                   '(or pass --eynollah-xml DIR with PAGE-XML made elsewhere)')

            model_dir = self.options.model_dir or eynollah_model_dir(log=self.log)
            self._runner = EynollahRunner(model_dir, self.options.policy(), log=self.log, warn=self.warn or self.log,
                                          progress=self.progress)
        return self._runner

    def xml_path(self, path: Path) -> Path:
        return self.xml_dir / f"{Path(path).stem}.xml"

    def prepare(self, paths: Sequence[Path]) -> RunResult:
        """Run eynollah on every image in ``paths`` that has no XML yet; returns the run's result."""
        self.last_result = self.runner.run([Path(p) for p in paths], self.xml_dir)
        return self.last_result

    def page(self, page: Page) -> EynollahPage:
        path = page.path
        if path is None:          # an in-memory page: give eynollah a file
            from PIL import Image

            path = self.xml_dir / f"page_{page.number:04d}.png"
            if not path.exists():
                Image.fromarray(page.image).save(path)
        path = Path(path)
        if path in self._pages:
            return self._pages[path]
        xml = self.xml_path(path)
        if not xml.is_file():
            self.prepare([path])
        if not xml.is_file():
            raise RuntimeError(f"eynollah produced no PAGE-XML for {path.name} (see {self.xml_dir / 'eynollah.log'})")
        parsed = parse_page_xml(xml)
        self._pages[path] = parsed
        return parsed

    def close(self) -> None:
        if self._temp is not None:
            shutil.rmtree(self._temp, ignore_errors=True)
            self._temp = self._xml_dir = None
