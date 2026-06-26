from __future__ import annotations

import re
from collections import defaultdict

from rapidfuzz import fuzz

from models.document import (
    Document,
    ExtractedField,
    OCRWord,
    Page,
)
from utils.logger import app_logger


class DocumentParsingError(Exception):
    """Raised when semantic document parsing fails."""


class DocumentParser:
    """
    Converts OCR, handwriting and checkbox detections into
    semantic document fields.
    """

    LABEL_THRESHOLD = 80

    LABEL_SEARCH_DISTANCE = 220

    LINE_TOLERANCE = 20

    def __init__(self) -> None:

        self.field_aliases = self._build_aliases()

    def parse_document(
        self,
        document: Document,
    ) -> Document:

        app_logger.info(
            "Parsing document..."
        )

        for page in document.pages:

            self.parse_page(page)

        app_logger.success(
            "Document parsing completed."
        )

        return document

    def parse_page(
        self,
        page: Page,
    ) -> None:

        try:

            printed_lines = self._build_lines(
                page.ocr_words
            )

            handwritten_lines = self._build_lines(
                page.handwriting_words
            )

            self._extract_label_value_pairs(
                page,
                printed_lines,
                handwritten_lines,
            )

            self._extract_checkbox_fields(
                page,
            )
            
            tables = self._extract_table_rows(
                printed_lines
            )
            
            self._table_rows_to_fields(
                page,
                tables,
            )
            
            self._resolve_duplicate_fields(
                page
            )

        except Exception as exc:

            app_logger.exception(
                exc
            )

            raise DocumentParsingError(
                f"Failed parsing page "
                f"{page.page_number}"
            ) from exc

    def _build_lines(
        self,
        words: list[OCRWord],
    ) -> list[list[OCRWord]]:

        if not words:
            return []

        ordered = sorted(

            words,

            key=lambda w: (

                min(p[1] for p in w.bbox),

                min(p[0] for p in w.bbox),

            ),
        )

        lines = []

        current = []

        current_y = None

        for word in ordered:

            y = sum(
                p[1]
                for p in word.bbox
            ) / 4

            if current_y is None:

                current.append(
                    word
                )

                current_y = y

                continue

            if abs(
                current_y - y
            ) <= self.LINE_TOLERANCE:

                current.append(
                    word
                )

            else:

                current.sort(

                    key=lambda w:

                    min(
                        p[0]
                        for p in w.bbox
                    )
                )

                lines.append(
                    current
                )

                current = [
                    word
                ]

                current_y = y

        if current:

            current.sort(

                key=lambda w:

                min(
                    p[0]
                    for p in w.bbox
                )
            )

            lines.append(
                current
            )

        return lines

    @staticmethod
    def _line_text(
        line,
    ) -> str:

        return " ".join(

            word.text.strip()

            for word in line

        ).strip()
        
    def _extract_label_value_pairs(
        self,
        page: Page,
        printed_lines: list[list[OCRWord]],
        handwritten_lines: list[list[OCRWord]],
    ) -> None:

        for printed in printed_lines:

            label = self._find_matching_label(
                self._line_text(
                    printed
                )
            )

            if label is None:
                continue

            value = self._find_handwritten_value(
                printed,
                handwritten_lines,
            )

            if value is None:

                value = self._extract_inline_value(
                    printed
                )

            if value is None:
                continue

            confidence = min(
                value.confidence,
                1.0,
            )

            page.add_field(

                ExtractedField(

                    name=label,

                    value=value.text,

                    confidence=confidence,

                    source=value.source,

                    page_number=page.page_number,

                    bbox=self._merge_bbox(
                        printed
                    ),
                )
            )

    def _extract_checkbox_fields(
        self,
        page: Page,
    ) -> None:

        for checkbox in page.checkboxes:

            if not checkbox.selected:
                continue

            if not checkbox.group:
                continue

            page.add_field(

                ExtractedField(

                    name=checkbox.group,

                    value=checkbox.value,

                    confidence=checkbox.confidence,

                    source="checkbox",

                    page_number=page.page_number,

                    bbox=checkbox.bbox,
                )
            )

    def _find_matching_label(
        self,
        text: str,
    ) -> str | None:

        normalized = text.lower().strip()

        best_label = None

        best_score = 0

        for label, aliases in self.field_aliases.items():

            for alias in aliases:

                score = fuzz.partial_ratio(
                    normalized,
                    alias,
                )

                if score > best_score:

                    best_score = score

                    best_label = label

        if best_score >= self.LABEL_THRESHOLD:
            return best_label

        return None

    def _find_handwritten_value(
        self,
        printed_line,
        handwritten_lines,
    ):

        if not handwritten_lines:
            return None

        px = max(

            p[0]

            for word in printed_line

            for p in word.bbox

        )

        py = sum(

            p[1]

            for word in printed_line

            for p in word.bbox

        ) / (

            len(printed_line) * 4

        )

        best = None

        distance = float("inf")

        for line in handwritten_lines:

            left = min(

                p[0]

                for word in line

                for p in word.bbox

            )

            y = sum(

                p[1]

                for word in line

                for p in word.bbox

            ) / (

                len(line) * 4

            )

            if abs(
                py - y
            ) > self.LINE_TOLERANCE:
                continue

            if left < px:
                continue

            d = left - px

            if d < distance:

                distance = d

                best = line

        if best is None:
            return None

        merged = OCRWord(

            text=self._line_text(
                best
            ),

            confidence=max(
                w.confidence
                for w in best
            ),

            bbox=self._merge_bbox(
                best
            ),

            source="handwritten",
        )

        return merged


    @staticmethod
    def _extract_inline_value(
        printed_line,
    ):

        text = " ".join(

            word.text

            for word in printed_line

        )

        if ":" not in text:
            return None

        label, value = text.split(
            ":",
            1,
        )

        value = value.strip()

        if not value:
            return None

        return OCRWord(

            text=value,

            confidence=1.0,

            bbox=[],

            source="printed",
        )

    def _build_blocks(
        self,
        lines: list[list[OCRWord]],
    ) -> list[list[list[OCRWord]]]:
        """
        Merge nearby text lines into semantic blocks.
        """

        if not lines:
            return []

        blocks = []
        current_block = [lines[0]]

        previous_bottom = self._line_bottom(lines[0])

        for line in lines[1:]:

            top = self._line_top(line)

            if top - previous_bottom <= 30:
                current_block.append(line)
            else:
                blocks.append(current_block)
                current_block = [line]

            previous_bottom = self._line_bottom(line)

        if current_block:
            blocks.append(current_block)

        return blocks

    def _merge_multiline_label(
        self,
        block: list[list[OCRWord]],
    ) -> str:

        text = []

        for line in block:

            text.append(
                self._line_text(line)
            )

        return " ".join(text).strip()

    def _extract_label_candidates(
        self,
        printed_lines: list[list[OCRWord]],
    ) -> list[tuple[str, list[OCRWord]]]:

        candidates = []

        blocks = self._build_blocks(
            printed_lines
        )

        for block in blocks:

            merged = self._merge_multiline_label(
                block
            )

            label = self._find_matching_label(
                merged
            )

            if label is not None:

                flattened = []

                for line in block:
                    flattened.extend(line)

                candidates.append(
                    (
                        label,
                        flattened,
                    )
                )

        return candidates

    def _extract_table_rows(
        self,
        printed_lines: list[list[OCRWord]],
    ) -> list[list[OCRWord]]:
        """
        Detect table rows using x-coordinate clustering.

        Government forms usually align table columns vertically,
        so we group words that share similar x positions.
        """

        table_rows: list[list[OCRWord]] = []

        previous_signature = None

        for line in printed_lines:

            if len(line) < 2:
                continue

            ordered = sorted(
                line,
                key=lambda w: min(
                    p[0] for p in w.bbox
                ),
            )

            gaps = []

            previous_right = None

            for word in ordered:

                left = min(
                    p[0] for p in word.bbox
                )

                right = max(
                    p[0] for p in word.bbox
                )

                if previous_right is not None:
                    gaps.append(
                        left - previous_right
                    )

                previous_right = right

            if not gaps:
                continue

            large_gaps = sum(
                gap > 45
                for gap in gaps
            )

            signature = tuple(
                round(
                    min(
                        p[0]
                        for p in word.bbox
                    ) / 50
                )
                for word in ordered
            )

            if (
                large_gaps >= 1
                and (
                    previous_signature is None
                    or signature == previous_signature
                )
            ):
                table_rows.append(
                    ordered
                )

                previous_signature = signature

        return table_rows
    
    def _table_rows_to_fields(
        self,
        page: Page,
        rows: list[list[OCRWord]],
    ) -> None:
        """
        Convert detected table rows into structured fields.
        """

        for index, row in enumerate(rows):

            text = self._line_text(
                row
            )

            if not text.strip():
                continue

            page.add_field(

                ExtractedField(

                    name=f"Table Row {index + 1}",

                    value=text,

                    confidence=min(
                        word.confidence
                        for word in row
                    ),

                    source="table",

                    page_number=page.page_number,

                    bbox=self._merge_bbox(
                        row
                    ),
                )
            )

    def _resolve_duplicate_fields(
        self,
        page: Page,
    ) -> None:

        unique = {}

        for field in page.extracted_fields:

            key = field.name.lower()

            if key not in unique:

                unique[key] = field

                continue

            if (
                field.confidence
                > unique[key].confidence
            ):
                unique[key] = field

        page.extracted_fields = list(
            unique.values()
        )

    @staticmethod
    def _merge_bbox(
        words: list[OCRWord],
    ) -> list[int]:

        xs = []
        ys = []

        for word in words:

            for point in word.bbox:

                xs.append(point[0])
                ys.append(point[1])

        return [

            min(xs),

            min(ys),

            max(xs),

            max(ys),

        ]

    @staticmethod
    def _line_top(
        line,
    ) -> int:

        return min(

            p[1]

            for word in line

            for p in word.bbox

        )

    @staticmethod
    def _line_bottom(
        line,
    ) -> int:

        return max(

            p[1]

            for word in line

            for p in word.bbox

        )