"""
Music score rendering pipeline.

High-level API::

    from src.rendering import render_prediction

    # Primus semantic tokens → sheet music image
    image = render_prediction(["clef-G2", "timeSignature-4/4", "note-C4_quarter", ...])

    # BeKern tokens → sheet music image
    image = render_prediction(bekern_tokens, format="bekern")

    # ABC notation string → sheet music image
    image = render_prediction(abc_string, format="abc")
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from src.rendering.interfaces import InputFormat, Score


def render_prediction(
    data: Union[List[str], str],
    format: str = "primus",
    **renderer_kwargs,
) -> Optional[np.ndarray]:
    """
    End-to-end rendering: model output → sheet music image.

    Args:
        data: Model output tokens (list of strings) or a raw notation string.
        format: One of ``"primus"``, ``"bekern"``, or ``"abc"``.
        **renderer_kwargs: Extra options forwarded to the renderer
            (e.g. ``page_width``, ``scale``).

    Returns:
        RGB image as numpy array (H, W, 3), or ``None`` on failure.
    """
    fmt = InputFormat(format)

    if fmt == InputFormat.PRIMUS:
        return _render_primus(data, **renderer_kwargs)  # type: ignore[arg-type]
    elif fmt == InputFormat.BEKERN:
        return _render_bekern(data, **renderer_kwargs)
    elif fmt == InputFormat.ABC:
        return _render_abc(data, **renderer_kwargs)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unsupported format: {format}")


def _render_primus(
    tokens: List[str], **renderer_kwargs
) -> Optional[np.ndarray]:
    """Primus semantic tokens → parse → MusicXML → render."""
    from src.rendering.converter import MusicXMLConverter
    from src.rendering.parser import PrimusParser
    from src.rendering.renderer import VerovioRenderer

    parser = PrimusParser()
    converter = MusicXMLConverter()
    renderer = VerovioRenderer(**renderer_kwargs)

    score: Score = parser.parse(tokens)
    musicxml: str = converter.convert(score)
    return renderer.render(musicxml)


def _render_bekern(
    data: Union[List[str], str], **renderer_kwargs
) -> Optional[np.ndarray]:
    """BeKern tokens → kern string → render."""
    from src.rendering.converter import BeKernConverter
    from src.rendering.renderer import VerovioRenderer

    converter = BeKernConverter()
    renderer = VerovioRenderer(**renderer_kwargs)

    kern_str: str = converter.convert(data)
    return renderer.render(kern_str)


def _render_abc(
    data: str, **renderer_kwargs
) -> Optional[np.ndarray]:
    """ABC notation → render directly."""
    from src.rendering.converter import ABCConverter
    from src.rendering.renderer import VerovioRenderer

    converter = ABCConverter()
    renderer = VerovioRenderer(**renderer_kwargs)

    abc_str: str = converter.convert(data)
    return renderer.render(abc_str)
