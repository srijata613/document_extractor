from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from models.document import Document


@dataclass(slots=True)
class ExtractionResult:
    """
    Final output produced by the extraction pipeline.
    """

    document: Document

    excel_rows: list[dict[str, Any]] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    warnings: list[str] = field(default_factory=list)

    errors: list[str] = field(default_factory=list)

    processing_time: float = 0.0

    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_excel_row(self, row: dict[str, Any]) -> None:
        self.excel_rows.append(row)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value