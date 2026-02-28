"""
Format converters: intermediate representation → notation strings.

Provides three converter implementations:
  * MusicXMLConverter – Score IR → MusicXML string  (Primus semantic path)
  * BeKernConverter   – bekern token list → kern string  (passthrough)
  * ABCConverter      – ABC string → ABC string  (passthrough)
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List, Union

from src.rendering.interfaces import (
    Barline,
    Clef,
    FormatConverter,
    KeySignature,
    Measure,
    MultiRest,
    MUSICXML_DIVISIONS,
    MusicElement,
    Note,
    Rest,
    Score,
    TimeSignature,
)


# ---------------------------------------------------------------------------
# MusicXMLConverter  (Primus path: Score IR → MusicXML)
# ---------------------------------------------------------------------------

class MusicXMLConverter(FormatConverter):
    """Convert a ``Score`` intermediate representation to a MusicXML string."""

    def convert(self, data: Score) -> str:  # type: ignore[override]
        measures = self._group_into_measures(data.elements)
        root = self._build_score_partwise(measures)

        ET.indent(root, space="  ")
        xml_str = ET.tostring(root, encoding="unicode", xml_declaration=False)
        return f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_str}'

    # -- measure grouping ---------------------------------------------------

    @staticmethod
    def _group_into_measures(elements: List[MusicElement]) -> List[Measure]:
        """Split the flat element list on ``Barline`` tokens into measures."""
        measures: list[Measure] = []
        current: list[MusicElement] = []

        for elem in elements:
            if isinstance(elem, Barline):
                if current:
                    measures.append(Measure(number=len(measures) + 1, elements=current))
                    current = []
            else:
                current.append(elem)

        # trailing elements after the last barline (or if no barlines)
        if current:
            measures.append(Measure(number=len(measures) + 1, elements=current))

        return measures

    # -- XML tree construction ----------------------------------------------

    def _build_score_partwise(self, measures: List[Measure]) -> ET.Element:
        root = ET.Element("score-partwise", version="4.0")

        # part-list
        part_list = ET.SubElement(root, "part-list")
        score_part = ET.SubElement(part_list, "score-part", id="P1")
        ET.SubElement(score_part, "part-name").text = "Music"

        # part
        part = ET.SubElement(root, "part", id="P1")
        for measure in measures:
            self._build_measure(part, measure)

        return root

    def _build_measure(self, parent: ET.Element, measure: Measure) -> None:
        m_elem = ET.SubElement(parent, "measure", number=str(measure.number))

        # Collect attribute elements (clef, key, time) at the start of the measure
        attr_elems: list[MusicElement] = []
        music_elems: list[MusicElement] = []
        for elem in measure.elements:
            if isinstance(elem, (Clef, KeySignature, TimeSignature)):
                attr_elems.append(elem)
            else:
                music_elems.append(elem)

        if attr_elems:
            attrs = ET.SubElement(m_elem, "attributes")
            ET.SubElement(attrs, "divisions").text = str(MUSICXML_DIVISIONS)
            for a in attr_elems:
                if isinstance(a, KeySignature):
                    key = ET.SubElement(attrs, "key")
                    ET.SubElement(key, "fifths").text = str(a.fifths)
                elif isinstance(a, TimeSignature):
                    time_el = ET.SubElement(attrs, "time")
                    if a.symbol:
                        time_el.set("symbol", a.symbol)
                    ET.SubElement(time_el, "beats").text = str(a.beats)
                    ET.SubElement(time_el, "beat-type").text = str(a.beat_type)
                elif isinstance(a, Clef):
                    clef = ET.SubElement(attrs, "clef")
                    ET.SubElement(clef, "sign").text = a.sign.value
                    ET.SubElement(clef, "line").text = str(a.line)
        elif measure.number == 1:
            # First measure always needs divisions even without explicit attributes
            attrs = ET.SubElement(m_elem, "attributes")
            ET.SubElement(attrs, "divisions").text = str(MUSICXML_DIVISIONS)

        for elem in music_elems:
            if isinstance(elem, Note):
                self._build_note(m_elem, elem)
            elif isinstance(elem, Rest):
                self._build_rest(m_elem, elem)
            elif isinstance(elem, MultiRest):
                self._build_multi_rest(m_elem, elem)

    def _build_note(self, parent: ET.Element, note: Note) -> None:
        n_elem = ET.SubElement(parent, "note")

        # Grace note marker (no <duration> for grace notes)
        if note.is_grace:
            ET.SubElement(n_elem, "grace")

        # Pitch
        pitch = ET.SubElement(n_elem, "pitch")
        ET.SubElement(pitch, "step").text = note.pitch.step
        if note.pitch.alter != 0:
            ET.SubElement(pitch, "alter").text = str(note.pitch.alter)
        ET.SubElement(pitch, "octave").text = str(note.pitch.octave)

        # Duration (omitted for grace notes)
        if not note.is_grace:
            dur_val = self._calc_duration(note.duration.quarter_length, note.dots)
            ET.SubElement(n_elem, "duration").text = str(dur_val)

        # Type
        ET.SubElement(n_elem, "type").text = note.duration.xml_type

        # Dots
        for _ in range(note.dots):
            ET.SubElement(n_elem, "dot")

        # Tie (sound-level)
        if note.is_tied_stop:
            ET.SubElement(n_elem, "tie", type="stop")
        if note.is_tied_start:
            ET.SubElement(n_elem, "tie", type="start")

        # Notations (visual ties, fermata)
        notations_needed = note.has_fermata or note.is_tied_start or note.is_tied_stop
        if notations_needed:
            notations = ET.SubElement(n_elem, "notations")
            if note.is_tied_stop:
                ET.SubElement(notations, "tied", type="stop")
            if note.is_tied_start:
                ET.SubElement(notations, "tied", type="start")
            if note.has_fermata:
                ET.SubElement(notations, "fermata")

    def _build_rest(self, parent: ET.Element, rest: Rest) -> None:
        n_elem = ET.SubElement(parent, "note")
        ET.SubElement(n_elem, "rest")

        dur_val = self._calc_duration(rest.duration.quarter_length, rest.dots)
        ET.SubElement(n_elem, "duration").text = str(dur_val)
        ET.SubElement(n_elem, "type").text = rest.duration.xml_type

        for _ in range(rest.dots):
            ET.SubElement(n_elem, "dot")

        if rest.has_fermata:
            notations = ET.SubElement(n_elem, "notations")
            ET.SubElement(notations, "fermata")

    @staticmethod
    def _build_multi_rest(parent: ET.Element, mrest: MultiRest) -> None:
        attrs = ET.SubElement(parent, "attributes")
        mr = ET.SubElement(attrs, "measure-style")
        ET.SubElement(mr, "multiple-rest").text = str(mrest.count)

    @staticmethod
    def _calc_duration(quarter_length: float, dots: int) -> int:
        """Calculate MusicXML <duration> value including dot multiplier."""
        base = quarter_length * MUSICXML_DIVISIONS
        if dots == 1:
            base *= 1.5
        elif dots == 2:
            base *= 1.75
        return int(base)


# ---------------------------------------------------------------------------
# BeKernConverter  (passthrough: bekern tokens → kern string)
# ---------------------------------------------------------------------------

class BeKernConverter(FormatConverter):
    """Convert bekern token list to a standard kern string for Verovio."""

    def convert(self, data: Union[List[str], str]) -> str:  # type: ignore[override]
        if isinstance(data, list):
            bekern_str = " ".join(data)
        else:
            bekern_str = data

        # Reuse the proven conversion logic from demos/demo_utils.py
        kern_str = bekern_str.replace("<b>", "\n")
        kern_str = kern_str.replace("<s>", " ")
        kern_str = kern_str.replace("<t>", "\t")
        kern_str = kern_str.replace("@", "")
        kern_str = kern_str.replace("·", "")

        if "**ekern" in kern_str:
            kern_str = kern_str.replace("**ekern", "**kern")
        elif not kern_str.startswith("**kern"):
            kern_str = "**kern\t**kern\n" + kern_str

        return kern_str


# ---------------------------------------------------------------------------
# ABCConverter  (passthrough)
# ---------------------------------------------------------------------------

class ABCConverter(FormatConverter):
    """Pass through ABC notation, adding minimal headers if missing.

    Verovio requires at least ``X:``, ``T:``, ``L:``, ``M:``, and ``K:``
    headers to recognise a string as ABC notation.  When the input comes
    from a ``strip_non_pitch`` pipeline only ``K:`` (and possibly ``V:``)
    survive, so we prepend sensible defaults for the missing fields.
    """

    def convert(self, data: str) -> str:  # type: ignore[override]
        return self._ensure_headers(data)

    @staticmethod
    def _ensure_headers(abc: str) -> str:
        """Add missing ABC headers so Verovio can parse the string."""
        lines = abc.split("\n")
        has = {tag: False for tag in ("X:", "T:", "M:", "L:", "K:")}
        for line in lines:
            for tag in has:
                if line.startswith(tag):
                    has[tag] = True

        if all(has.values()):
            return abc  # nothing to add

        # Build header block with sensible defaults for missing fields
        header: list[str] = []
        if not has["X:"]:
            header.append("X:1")
        if not has["T:"]:
            header.append("T:Untitled")
        if not has["M:"]:
            header.append("M:4/4")
        if not has["L:"]:
            header.append("L:1/8")

        # Insert headers before the first line, keeping K: in its
        # original position (it must appear last among headers).
        return "\n".join(header) + "\n" + abc
