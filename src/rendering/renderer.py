"""
Verovio-based score renderer.

Implements the ``ScoreRenderer`` interface using Verovio for notation
engraving and CairoSVG for SVG-to-PNG rasterization.  Supports MusicXML,
kern, and ABC notation strings — Verovio auto-detects the format.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.rendering.interfaces import ScoreRenderer

logger = logging.getLogger(__name__)

try:
    import verovio

    _VEROVIO_AVAILABLE = True
except ImportError:
    _VEROVIO_AVAILABLE = False

try:
    import cairosvg

    _CAIROSVG_AVAILABLE = True
except ImportError:
    _CAIROSVG_AVAILABLE = False

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


class VerovioRenderer(ScoreRenderer):
    """
    Render notation strings to images via Verovio + CairoSVG.

    The renderer lazily initialises a ``verovio.toolkit`` on first use and
    reuses it across calls for performance.

    Parameters
    ----------
    page_width : int
        Page width in Verovio abstract units (default matches demo_utils).
    scale : int
        Scaling percentage (default 40, matching demo_utils).
    adjust_page_height : bool
        Let Verovio shrink the page height to fit the content.
    transpose : int
        Number of semitones to transpose (0 = no transposition).
    """

    def __init__(
        self,
        page_width: int = 2100,
        scale: int = 40,
        adjust_page_height: bool = True,
        transpose: int = 0,
    ) -> None:
        self._page_width = page_width
        self._scale = scale
        self._adjust_page_height = adjust_page_height
        self._transpose = transpose
        self._tk: Optional[verovio.toolkit] = None  # type: ignore[name-defined]

    # -- lazy init -------------------------------------------------------------

    def _ensure_toolkit(self) -> bool:
        """Initialise the Verovio toolkit if not already done.

        Returns ``True`` when the toolkit is ready, ``False`` otherwise.
        """
        if self._tk is not None:
            return True
        if not _VEROVIO_AVAILABLE:
            logger.warning("verovio is not installed — rendering disabled")
            return False

        self._tk = verovio.toolkit()
        options: dict = {
            "pageWidth": self._page_width,
            "scale": self._scale,
            "adjustPageHeight": self._adjust_page_height,
            "footer": "none",
            "header": "none",
        }
        if self._transpose != 0:
            options["transpose"] = str(self._transpose)
        self._tk.setOptions(options)
        return True

    # -- SVG rendering ---------------------------------------------------------

    def _render_page_svg(self, page: int = 1) -> Optional[str]:
        """Render a single page to an SVG string."""
        assert self._tk is not None
        svg = self._tk.renderToSVG(page)
        if not svg or not svg.strip():
            return None
        return svg

    # -- SVG → numpy ----------------------------------------------------------

    @staticmethod
    def _svg_to_numpy(svg: str) -> Optional[np.ndarray]:
        """Convert an SVG string to an RGB numpy array via CairoSVG + OpenCV."""
        if not _CAIROSVG_AVAILABLE:
            logger.warning("cairosvg is not installed — PNG conversion disabled")
            return None
        if not _CV2_AVAILABLE:
            logger.warning("opencv (cv2) is not installed — PNG decoding disabled")
            return None

        png_bytes = cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            background_color="white",
        )
        arr = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return None
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    # -- public API (ScoreRenderer) --------------------------------------------

    def render(self, notation_str: str) -> Optional[np.ndarray]:
        """Render a notation string (MusicXML / kern / ABC) to an RGB image.

        Multi-page scores are vertically concatenated into a single image.

        Returns ``(H, W, 3)`` numpy array or ``None`` on failure.
        """
        if not self._ensure_toolkit():
            return None

        assert self._tk is not None

        try:
            if not self._tk.loadData(notation_str):
                logger.error("Verovio failed to load notation data")
                return None

            page_count = self._tk.getPageCount()
            if page_count == 0:
                logger.error("Verovio produced zero pages")
                return None

            pages: list[np.ndarray] = []
            for page_num in range(1, page_count + 1):
                svg = self._render_page_svg(page_num)
                if svg is None:
                    logger.warning("Empty SVG for page %d/%d", page_num, page_count)
                    continue
                img = self._svg_to_numpy(svg)
                if img is None:
                    logger.warning(
                        "Failed to rasterise page %d/%d", page_num, page_count
                    )
                    continue
                pages.append(img)

            if not pages:
                logger.error("No pages were successfully rendered")
                return None

            if len(pages) == 1:
                return pages[0]

            # Vertically concatenate pages, padding narrower ones with white
            max_width = max(p.shape[1] for p in pages)
            padded = []
            for p in pages:
                if p.shape[1] < max_width:
                    pad = np.full(
                        (p.shape[0], max_width - p.shape[1], 3),
                        255,
                        dtype=np.uint8,
                    )
                    p = np.concatenate([p, pad], axis=1)
                padded.append(p)
            return np.concatenate(padded, axis=0)

        except Exception:
            logger.exception("Unexpected error during rendering")
            return None

    def render_to_svg(self, notation_str: str) -> Optional[str]:
        """Render to raw SVG string (first page only)."""
        if not self._ensure_toolkit():
            return None

        assert self._tk is not None

        try:
            if not self._tk.loadData(notation_str):
                logger.error("Verovio failed to load notation data")
                return None
            return self._render_page_svg(1)
        except Exception:
            logger.exception("Unexpected error during SVG rendering")
            return None
