"""
Rendering pipeline interfaces and shared data model.

Defines the intermediate representation (IR) used between the parser and converter,
plus abstract base classes for each pipeline stage:

    semantic labels → Parser → IR (dataclasses) → Converter → format string → Renderer → image

Three supported input formats:
  1. Primus semantic tokens  → parsed to IR → converted to MusicXML → Verovio
  2. BeKern tokens           → converted to kern string directly   → Verovio
  3. ABC notation            → passed through directly             → Verovio
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InputFormat(enum.Enum):
    """Supported model output formats."""
    PRIMUS = "primus"
    BEKERN = "bekern"
    ABC = "abc"


class ClefSign(enum.Enum):
    G = "G"
    F = "F"
    C = "C"


class Accidental(enum.Enum):
    """
    Accidental modifiers for pitches.

    In the Primus vocabulary only SHARP (#), FLAT (b), and NATURAL (implicit)
    actually occur.  DOUBLE_SHARP / DOUBLE_FLAT are kept for completeness.
    """
    SHARP = 1
    FLAT = -1
    DOUBLE_SHARP = 2
    DOUBLE_FLAT = -2
    NATURAL = 0


class Duration(enum.Enum):
    """Note/rest durations with their MusicXML type names and quarter-note multiples."""
    QUADRUPLE_WHOLE = ("long", 16.0)
    DOUBLE_WHOLE = ("breve", 8.0)
    WHOLE = ("whole", 4.0)
    HALF = ("half", 2.0)
    QUARTER = ("quarter", 1.0)
    EIGHTH = ("eighth", 0.5)
    SIXTEENTH = ("16th", 0.25)
    THIRTY_SECOND = ("32nd", 0.125)
    SIXTY_FOURTH = ("64th", 0.0625)
    HUNDRED_TWENTY_EIGHTH = ("128th", 0.03125)

    def __init__(self, xml_type: str, quarter_length: float):
        self.xml_type = xml_type
        self.quarter_length = quarter_length


# Mapping from Primus duration strings to Duration enum
DURATION_MAP: dict[str, Duration] = {
    "quadruple_whole": Duration.QUADRUPLE_WHOLE,
    "double_whole": Duration.DOUBLE_WHOLE,
    "whole": Duration.WHOLE,
    "half": Duration.HALF,
    "quarter": Duration.QUARTER,
    "eighth": Duration.EIGHTH,
    "sixteenth": Duration.SIXTEENTH,
    "thirty_second": Duration.THIRTY_SECOND,
    "sixty_fourth": Duration.SIXTY_FOURTH,
    "hundred_twenty_eighth": Duration.HUNDRED_TWENTY_EIGHTH,
}

# MusicXML divisions per quarter note (all durations are integers at this value)
MUSICXML_DIVISIONS = 32

# Primus accidental character → Accidental enum
ACCIDENTAL_MAP: dict[str, Accidental] = {
    "#": Accidental.SHARP,
    "b": Accidental.FLAT,
}

# Key signature → MusicXML <fifths> value
KEY_SIGNATURE_FIFTHS: dict[str, int] = {
    "CM": 0,
    "GM": 1,
    "DM": 2,
    "AM": 3,
    "EM": 4,
    "BM": 5,
    "F#M": 6,
    "C#M": 7,
    "FM": -1,
    "BbM": -2,
    "EbM": -3,
    "AbM": -4,
    "DbM": -5,
    "GbM": -6,
}


# ---------------------------------------------------------------------------
# Intermediate Representation – dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Pitch:
    """A musical pitch."""
    step: str          # A-G
    octave: int        # 1-8
    alter: int = 0     # -2, -1, 0, 1, 2 (flats/sharps)


@dataclass
class Note:
    """
    A sounding note (or grace note when ``is_grace=True``).

    Primus token parsing order for the detail part after the pitch::

        1. Strip ``_fermata`` or ``._fermata`` suffix → has_fermata
        2. Count and strip trailing dots (``.`` / ``..``) → dots
        3. Look up remaining string in DURATION_MAP → duration

    Tie handling: the standalone ``tie`` token sets ``is_tied_start`` on the
    preceding note and ``is_tied_stop`` on the following note.
    """
    pitch: Pitch
    duration: Duration
    dots: int = 0             # 0, 1 (dotted), or 2 (double-dotted)
    is_grace: bool = False
    has_fermata: bool = False
    is_tied_start: bool = False
    is_tied_stop: bool = False


@dataclass
class Rest:
    """A rest."""
    duration: Duration
    dots: int = 0
    has_fermata: bool = False


@dataclass
class Clef:
    """A clef indication."""
    sign: ClefSign
    line: int  # staff line number


@dataclass
class KeySignature:
    """A key signature."""
    fifths: int  # -7..+7, negative = flats, positive = sharps


@dataclass
class TimeSignature:
    """A time signature."""
    beats: int
    beat_type: int
    symbol: Optional[str] = None  # "common" or "cut" for C and C/ signatures


@dataclass
class Barline:
    """A barline (marks measure boundary in the flat element list)."""
    pass


# Special time signatures (not in {num}/{den} format)
COMMON_TIME = TimeSignature(beats=4, beat_type=4, symbol="common")   # timeSignature-C
CUT_TIME = TimeSignature(beats=2, beat_type=2, symbol="cut")         # timeSignature-C/


@dataclass
class MultiRest:
    """A multi-measure rest."""
    count: int


# Union of all musical elements that can appear in a measure
MusicElement = Note | Rest | Clef | KeySignature | TimeSignature | Barline | MultiRest


@dataclass
class Measure:
    """A single measure containing a sequence of music elements."""
    number: int
    elements: List[MusicElement] = field(default_factory=list)


@dataclass
class Score:
    """
    Top-level intermediate representation of a parsed score.

    For the Primus (monophonic) path the score is a flat sequence of
    ``MusicElement`` objects that the converter will group into measures
    based on ``Barline`` boundaries.
    """
    elements: List[MusicElement] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract interfaces for pipeline stages
# ---------------------------------------------------------------------------

class SemanticParser(ABC):
    """
    Stage 1 – Parse raw model output tokens into the intermediate
    representation.

    Implementations:
      * ``PrimusParser``  – parses Primus semantic tokens into a ``Score``
    """

    @abstractmethod
    def parse(self, tokens: List[str]) -> Score:
        """
        Parse a list of semantic token strings into a Score IR.

        Args:
            tokens: Decoded semantic label strings
                    (e.g. ``["clef-G2", "note-C4_quarter", ...]``).

        Returns:
            A ``Score`` containing the parsed ``MusicElement`` sequence.
        """
        ...


class FormatConverter(ABC):
    """
    Stage 2 – Convert the intermediate representation (or raw tokens)
    into a notation string consumable by the renderer.

    Implementations:
      * ``MusicXMLConverter`` – IR ``Score`` → MusicXML string (Primus path)
      * ``BeKernConverter``   – raw bekern tokens → kern string (passthrough)
      * ``ABCConverter``      – raw ABC string → ABC string (passthrough)

    MusicXML generation notes (for ``MusicXMLConverter``):
      * Use ``divisions=32`` per quarter note so all durations are integers.
      * ``<duration>`` must reflect actual time including dots
        (e.g. dotted quarter = 48, not 32).  ``<dot/>`` is for rendering only.
      * Grace notes: emit ``<grace/>`` inside ``<note>``, include ``<type>``
        but **no** ``<duration>`` element.
      * Ties require **both** ``<tie type="start"/>`` (in ``<note>``) and
        ``<tied type="start"/>`` (in ``<notations>``) for correct Verovio
        rendering.
      * Include ``<?xml version="1.0" encoding="UTF-8"?>`` declaration.
      * Mid-piece clef/key/time changes go in a new ``<attributes>`` block.
    """

    @abstractmethod
    def convert(self, data) -> str:
        """
        Convert parsed data into a notation format string.

        Args:
            data: Either a ``Score`` object (Primus) or raw token list /
                  string (BeKern / ABC).

        Returns:
            A notation string (MusicXML, kern, or ABC).
        """
        ...


class ScoreRenderer(ABC):
    """
    Stage 3 – Render a notation string to a raster image.

    Implementations:
      * ``VerovioRenderer`` – renders via the Verovio toolkit
    """

    @abstractmethod
    def render(self, notation_str: str) -> Optional[np.ndarray]:
        """
        Render a notation string to an image.

        Args:
            notation_str: MusicXML, kern, or ABC notation string.

        Returns:
            RGB image as a numpy array (H, W, 3), or ``None`` on failure.
        """
        ...

    def render_to_svg(self, notation_str: str) -> Optional[str]:
        """
        Render a notation string to an SVG string (optional).

        Subclasses may override this for file-export workflows.
        The default implementation returns ``None``.
        """
        return None
