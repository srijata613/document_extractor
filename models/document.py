from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PageAssets:
    """
    Stores every intermediate image generated while processing one page.
    """

    original: Path

    preprocessed: Path | None = None

    threshold: Path | None = None

    ocr_image: Path | None = None

    handwriting_image: Path | None = None

    checkbox_image: Path | None = None

    debug_image: Path | None = None


@dataclass(slots=True)
class OCRWord:
    """
    Represents one OCR prediction.
    """

    text: str

    confidence: float

    bbox: list[list[int]]

    source: str = "printed"


@dataclass(slots=True)
class Checkbox:
    """
    Represents one detected checkbox.
    """

    label: str

    checked: bool

    confidence: float

    bbox: list[int]


@dataclass(slots=True)
class Page:
    """
    Represents one page inside a document.
    """

    page_number: int

    width: int

    height: int

    rotation: int

    assets: PageAssets

    ocr_words: list[OCRWord] = field(default_factory=list)

    handwriting_words: list[OCRWord] = field(default_factory=list)

    checkboxes: list[Checkbox] = field(default_factory=list)

    extracted_fields: dict[str, Any] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    def add_ocr_word(self, word: OCRWord) -> None:
        self.ocr_words.append(word)

    def add_handwriting_word(self, word: OCRWord) -> None:
        self.handwriting_words.append(word)

    def add_checkbox(self, checkbox: Checkbox) -> None:
        self.checkboxes.append(checkbox)

    def add_field(self, key: str, value: Any) -> None:
        self.extracted_fields[key] = value

    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value


@dataclass(slots=True)
class Document:
    """
    Represents one uploaded PDF.
    """

    file_path: Path

    file_name: str

    pages: list[Page] = field(default_factory=list)

    document_type: str = "Unknown"

    metadata: dict[str, Any] = field(default_factory=dict)

    def add_page(self, page: Page) -> None:
        self.pages.append(page)

    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def is_empty(self) -> bool:
        return self.total_pages == 0