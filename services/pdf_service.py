from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import fitz

from config import (
    PDF_RENDER_DPI,
    RENDERED_DIR,
)
from models.document import (
    Document,
    Page,
    PageAssets,
)
from utils.logger import app_logger


class PDFProcessingError(Exception):
    """Raised when PDF processing fails."""


class PDFService:
    """
    Production PDF rendering service.

    Responsibilities
    ----------------
    • Validate PDF
    • Open safely
    • Render every page
    • Create working directory
    • Build Document object
    • Cleanup on failure
    """

    def __init__(self) -> None:

        zoom = PDF_RENDER_DPI / 72

        self.matrix = fitz.Matrix(zoom, zoom)


    def load_document(
        self,
        pdf_path: Path,
    ) -> Document:

        self._validate(pdf_path)

        working_dir = self._create_working_directory(pdf_path)

        app_logger.info(
            f"Processing {pdf_path.name}"
        )

        try:

            with fitz.open(pdf_path) as pdf:

                self._unlock_if_needed(pdf)

                document = Document(
                    file_path=pdf_path,
                    file_name=pdf_path.name,
                )

                document.add_metadata(
                    "working_directory",
                    (working_dir),
                )

                document.add_metadata(
                    "page_count",
                    len(pdf),
                )

                document.add_metadata(
                    "pdf_version",
                    pdf.metadata.get(
                        "format",
                        "",
                    ),
                )

                for page_index in range(len(pdf)):

                    page = self._render_page(
                        pdf,
                        page_index,
                        pdf_path.stem,
                        working_dir,
                    )

                    document.add_page(page)

                app_logger.success(
                    f"Rendered {document.total_pages} pages."
                )

                return document

        except Exception as exc:

            self.cleanup(working_dir)

            app_logger.exception(exc)

            raise PDFProcessingError(
                "Failed to process PDF."
            ) from exc


    def _validate(
        self,
        pdf_path: Path,
    ) -> None:

        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        if not pdf_path.is_file():
            raise PDFProcessingError(
                "Invalid PDF."
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise PDFProcessingError(
                "Only PDF files are supported."
            )

        if pdf_path.stat().st_size == 0:
            raise PDFProcessingError(
                "Empty PDF."
            )


    @staticmethod
    def _unlock_if_needed(
        pdf: fitz.Document,
    ) -> None:

        if not pdf.is_encrypted:
            return

        if pdf.authenticate(""):
            return

        raise PDFProcessingError(
            "Password protected PDF."
        )


    @staticmethod
    def _create_working_directory(
        pdf_path: Path,
    ) -> Path:

        directory = (
            RENDERED_DIR
            / f"{pdf_path.stem}_{uuid4().hex}"
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return directory


    def _render_page(
        self,
        pdf: fitz.Document,
        page_index: int,
        stem: str,
        working_dir: Path,
    ) -> Page:

        pdf_page = pdf.load_page(page_index)

        pixmap = pdf_page.get_pixmap(
            matrix=self.matrix,
            alpha=False,
        )

        image_path = (
            working_dir
            / f"{stem}_page_{page_index + 1}.png"
        )

        pixmap.save(image_path)

        assets = PageAssets(
            original=image_path,
        )

        page = Page(
            page_number=page_index + 1,
            width=pixmap.width,
            height=pixmap.height,
            rotation=pdf_page.rotation,
            assets=assets,
        )

        page.add_metadata(
            "dpi",
            PDF_RENDER_DPI,
        )

        page.add_metadata(
            "rect_width",
            pdf_page.rect.width,
        )

        page.add_metadata(
            "rect_height",
            pdf_page.rect.height,
        )

        page.add_metadata(
            "xref",
            pdf_page.xref,
        )

        app_logger.info(
            f"Rendered page {page.page_number}"
        )

        return page


    @staticmethod
    def cleanup(
        working_directory: Path,
    ) -> None:

        if not working_directory.exists():
            return

        try:

            shutil.rmtree(
                working_directory,
                ignore_errors=True,
            )

            app_logger.info(
                "Temporary files cleaned."
            )

        except Exception as exc:

            app_logger.warning(
                f"Cleanup failed: {exc}"
            )


pdf_service = PDFService()