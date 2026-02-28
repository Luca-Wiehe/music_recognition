"""
Primus semantic token parser.

Parses CTC-decoded semantic label sequences from the MonophonicModel (CRNN)
into the intermediate ``Score`` representation defined in ``interfaces.py``.

Token format reference (from ``data/semantic_labels.txt``):
  - ``barline``
  - ``clef-{sign}{line}``          e.g. ``clef-G2``
  - ``gracenote-{pitch}_{dur}``    e.g. ``gracenote-C#5_eighth``
  - ``keySignature-{key}M``        e.g. ``keySignature-AbM``
  - ``multirest-{count}``          e.g. ``multirest-4``
  - ``note-{pitch}_{dur}[mods]``   e.g. ``note-A4_quarter._fermata``
  - ``rest-{dur}[mods]``           e.g. ``rest-eighth.``
  - ``tie``
  - ``timeSignature-{n}/{d}``      e.g. ``timeSignature-3/4``
  - ``timeSignature-C``  / ``timeSignature-C/``
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from .interfaces import (
    ACCIDENTAL_MAP,
    COMMON_TIME,
    CUT_TIME,
    DURATION_MAP,
    KEY_SIGNATURE_FIFTHS,
    Accidental,
    Barline,
    Clef,
    ClefSign,
    Duration,
    KeySignature,
    MusicElement,
    MultiRest,
    Note,
    Pitch,
    Rest,
    Score,
    SemanticParser,
    TimeSignature,
)

logger = logging.getLogger(__name__)

# Regex for extracting pitch components: note name, optional accidental, octave
_PITCH_RE = re.compile(r"^([A-G])(#|b)?(\d+)$")


class PrimusParser(SemanticParser):
    """
    Parses a list of Primus semantic tokens into a ``Score``.

    The parser is stateful to handle ``tie`` tokens that link the preceding
    note to the following note.
    """

    def parse(self, tokens: List[str]) -> Score:
        """
        Parse Primus semantic tokens into a Score IR.

        Args:
            tokens: Decoded semantic label strings from CTC output,
                    e.g. ``["clef-G2", "timeSignature-4/4", "note-C4_quarter", ...]``.

        Returns:
            A ``Score`` with a flat list of ``MusicElement`` objects.
        """
        elements: List[MusicElement] = []
        pending_tie_stop = False

        for token in tokens:
            parsed = self._parse_token(token, elements, pending_tie_stop)

            if parsed is not None:
                elements.append(parsed)
                # If there was a pending tie stop and we just added a Note,
                # clear the flag.
                if pending_tie_stop and isinstance(parsed, Note):
                    pending_tie_stop = False

            # Handle tie: mark preceding note and flag next note
            if token == "tie":
                # Find the most recent Note in elements and mark it
                for elem in reversed(elements):
                    if isinstance(elem, Note):
                        elem.is_tied_start = True
                        break
                pending_tie_stop = True

        return Score(elements=elements)

    def _parse_token(
        self,
        token: str,
        elements: List[MusicElement],
        pending_tie_stop: bool,
    ) -> Optional[MusicElement]:
        """Parse a single semantic token into a MusicElement, or None if skipped."""

        if token == "barline":
            return Barline()

        if token == "tie":
            # Handled separately in parse() for stateful tie logic
            return None

        # Split on first '-' to get category
        if "-" not in token:
            logger.warning("Unknown token (no category separator): %s", token)
            return None

        category, detail = token.split("-", 1)

        if category == "clef":
            return self._parse_clef(detail)
        elif category == "keySignature":
            return self._parse_key_signature(detail)
        elif category == "timeSignature":
            return self._parse_time_signature(detail)
        elif category == "multirest":
            return self._parse_multirest(detail)
        elif category == "note":
            return self._parse_note(detail, is_grace=False, pending_tie_stop=pending_tie_stop)
        elif category == "gracenote":
            return self._parse_note(detail, is_grace=True, pending_tie_stop=pending_tie_stop)
        elif category == "rest":
            return self._parse_rest(detail)
        else:
            logger.warning("Unknown token category '%s' in token: %s", category, token)
            return None

    # ------------------------------------------------------------------
    # Individual parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_clef(detail: str) -> Optional[Clef]:
        """Parse clef detail like ``G2``, ``F4``, ``C3``."""
        if len(detail) < 2:
            logger.warning("Invalid clef detail: %s", detail)
            return None

        sign_char = detail[0]
        try:
            sign = ClefSign(sign_char)
        except ValueError:
            logger.warning("Unknown clef sign: %s", sign_char)
            return None

        try:
            line = int(detail[1:])
        except ValueError:
            logger.warning("Invalid clef line number: %s", detail[1:])
            return None

        return Clef(sign=sign, line=line)

    @staticmethod
    def _parse_key_signature(detail: str) -> Optional[KeySignature]:
        """Parse key signature detail like ``CM``, ``AbM``, ``F#M``."""
        fifths = KEY_SIGNATURE_FIFTHS.get(detail)
        if fifths is None:
            logger.warning("Unknown key signature: %s", detail)
            return None
        return KeySignature(fifths=fifths)

    @staticmethod
    def _parse_time_signature(detail: str) -> Optional[TimeSignature]:
        """Parse time signature detail like ``4/4``, ``C``, ``C/``."""
        # Special cases
        if detail == "C":
            return TimeSignature(
                beats=COMMON_TIME.beats,
                beat_type=COMMON_TIME.beat_type,
                symbol=COMMON_TIME.symbol,
            )
        if detail == "C/":
            return TimeSignature(
                beats=CUT_TIME.beats,
                beat_type=CUT_TIME.beat_type,
                symbol=CUT_TIME.symbol,
            )

        # General case: {num}/{den}
        parts = detail.split("/")
        if len(parts) != 2:
            logger.warning("Invalid time signature format: %s", detail)
            return None

        try:
            beats = int(parts[0])
            beat_type = int(parts[1])
        except ValueError:
            logger.warning("Non-integer time signature values: %s", detail)
            return None

        return TimeSignature(beats=beats, beat_type=beat_type)

    @staticmethod
    def _parse_multirest(detail: str) -> Optional[MultiRest]:
        """Parse multirest detail like ``4``, ``32``."""
        try:
            count = int(detail)
        except ValueError:
            logger.warning("Invalid multirest count: %s", detail)
            return None
        return MultiRest(count=count)

    def _parse_note(
        self, detail: str, *, is_grace: bool, pending_tie_stop: bool
    ) -> Optional[Note]:
        """
        Parse note/gracenote detail like ``C#5_eighth``, ``A4_quarter._fermata``.

        The detail part has the structure: ``{pitch}_{duration}[.][._fermata]``
        """
        # Split pitch from duration at the first underscore
        underscore_idx = detail.find("_")
        if underscore_idx == -1:
            logger.warning("No duration separator in note: %s", detail)
            return None

        pitch_str = detail[:underscore_idx]
        duration_str = detail[underscore_idx + 1 :]

        # Parse pitch
        pitch = self._parse_pitch(pitch_str)
        if pitch is None:
            return None

        # Parse duration with modifiers
        duration, dots, has_fermata = self._parse_duration_modifiers(duration_str)
        if duration is None:
            return None

        return Note(
            pitch=pitch,
            duration=duration,
            dots=dots,
            is_grace=is_grace,
            has_fermata=has_fermata,
            is_tied_stop=pending_tie_stop,
        )

    def _parse_rest(self, detail: str) -> Optional[Rest]:
        """Parse rest detail like ``quarter``, ``eighth.``, ``half_fermata``."""
        duration, dots, has_fermata = self._parse_duration_modifiers(detail)
        if duration is None:
            return None

        return Rest(duration=duration, dots=dots, has_fermata=has_fermata)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_pitch(pitch_str: str) -> Optional[Pitch]:
        """
        Parse a pitch string like ``C4``, ``Ab5``, ``F#3``.

        Pattern: ``([A-G])(#|b)?(\\d+)``
        """
        match = _PITCH_RE.match(pitch_str)
        if not match:
            logger.warning("Invalid pitch: %s", pitch_str)
            return None

        step = match.group(1)
        accidental_char = match.group(2)
        octave = int(match.group(3))

        alter = 0
        if accidental_char is not None:
            acc = ACCIDENTAL_MAP.get(accidental_char)
            if acc is not None:
                alter = acc.value

        return Pitch(step=step, octave=octave, alter=alter)

    @staticmethod
    def _parse_duration_modifiers(dur_str: str) -> tuple[Optional[Duration], int, bool]:
        """
        Parse a duration string that may include dots and/or fermata.

        Parsing order:
          1. Strip ``_fermata`` suffix → has_fermata
          2. Count and strip trailing dots → dots
          3. Look up remaining string in DURATION_MAP → Duration

        Args:
            dur_str: e.g. ``quarter``, ``eighth.``, ``half._fermata``,
                     ``quarter.._fermata``

        Returns:
            Tuple of (Duration or None, dot count, has_fermata).
        """
        has_fermata = False
        dots = 0

        s = dur_str

        # Step 1: strip fermata suffix (handles both _fermata and ._fermata)
        if s.endswith("_fermata"):
            has_fermata = True
            s = s[: -len("_fermata")]
            # Handle ._fermata: the dot before _fermata is a duration dot
            # It will be counted in step 2

        # Step 2: count and strip trailing dots
        while s.endswith("."):
            dots += 1
            s = s[:-1]

        # Step 3: look up duration
        duration = DURATION_MAP.get(s)
        if duration is None:
            logger.warning("Unknown duration: '%s' (from '%s')", s, dur_str)
            return None, 0, False

        return duration, dots, has_fermata
