from __future__ import annotations

from dataclasses import fields
import time
from pathlib import Path
from typing import Any, Callable

from models.document import (
    Document,
    MappedField,
)

from services.checkbox_detector import checkbox_detector
from services.document_parser import DocumentParser
from services.excel_exporter import excel_exporter
from services.field_mapper import FieldMapper
from services.handwriting import handwriting_service
from services.ocr_service import ocr_service
from services.pdf_service import pdf_service
from services.preprocess import preprocess_service

from utils.logger import app_logger


class PipelineError(Exception):
    """Raised when document pipeline fails."""


class DocumentPipeline:
    """
    Production document extraction pipeline.

    Responsibilities
    ----------------
    1. Load PDF
    2. Preprocess pages
    3. OCR
    4. Handwriting Recognition
    5. Checkbox Detection
    6. Document Parsing
    7. Field Mapping
    8. Excel Export
    """

    def __init__(self) -> None:

        self.parser = DocumentParser()

        self.mapper = FieldMapper()

    def process(
        self,
        input_file: Path,
        output_directory: Path,
        output_filename: str | None =  None,
    ) -> Path:

        app_logger.info(
            f"Processing document: {input_file.name}"
        )

        if not input_file.exists():

            raise PipelineError(
                f"Input file not found: {input_file}"
            )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        document: Document | None = None

        try:

            document = self._run_stage(
                "PDF Loading",
                self._load_document,
                input_file,
            )

            self._run_stage(
                "Preprocessing",
                self._preprocess,
                document,
            )

            self._run_stage(
                "Printed OCR",
                self._run_ocr,
                document,
            )

            self._run_stage(
                "Handwriting OCR",
                self._run_handwriting,
                document,
            )

            self._run_stage(
                "Checkbox Detection",
                self._run_checkbox_detection,
                document,
            )

            self._run_stage(
                "Document Parsing",
                self._parse_document,
                document,
            )

            mapped_fields = self._run_stage(
                "Field Mapping",
                self._map_fields,
                document,
            )

            output_file = self._run_stage(
                "Excel Export",
                excel_exporter.export,
                document,
                mapped_fields,
                output_directory,
                output_filename or input_file.stem,
            )

            if not output_file.exists():

                raise PipelineError(
                    "Excel export failed."
                )

            self.cleanup(
                document
            )

            app_logger.success(
                f"Pipeline completed successfully.\n"
                f"Output: {output_file}"
            )

            return output_file

        except Exception as exc:

            if document is not None:

                self.cleanup(
                    document
                )

            app_logger.exception(
                exc
            )

            raise PipelineError(
                str(exc)
            ) from exc

    def _run_stage(
        self,
        stage_name: str,
        function: Callable[..., Any],
        *args,
        **kwargs,
    ) -> Any:

        app_logger.info(
            f"Starting {stage_name}..."
        )

        start = time.perf_counter()

        result = function(
            *args,
            **kwargs,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        app_logger.success(
            f"{stage_name} completed "
            f"in {elapsed:.2f}s"
        )

        return result

    @staticmethod
    def _load_document(
        path: Path,
    ) -> Document:

        document = pdf_service.load_document(
            path
        )

        DocumentPipeline.validate_document(
            document
        )

        return document

    @staticmethod
    def _preprocess(
        document: Document,
    ) -> None:

        preprocess_service.process_document(
            document
        )

    @staticmethod
    def _run_ocr(
        document: Document,
    ) -> None:

        ocr_service.process_document(
            document
        )

    @staticmethod
    def _run_handwriting(
        document: Document,
    ) -> None:

        handwriting_service.process_document(
            document
        )
        
    @staticmethod
    def _run_checkbox_detection(
        document: Document,
    ) -> None:

        checkbox_detector.process_document(
            document
        )

    def _parse_document(
        self,
        document: Document,
    ) -> None:

        self.parser.parse_document(
            document
        )

    def _map_fields(
        self,
        document: Document,
    ) -> dict[str, MappedField]:

        return self.mapper.map_document(
            document
        )

    @staticmethod
    def validate_document(
        document: Document,
    ) -> None:
        """
        Validate the document before processing.
        """

        if not document.pages:

            raise PipelineError(
                "Document contains no pages."
            )

        for page in document.pages:

            if page.assets is None:

                raise PipelineError(
                    f"Page {page.page_number} has no assets."
                )

            if page.assets.original is None:

                raise PipelineError(
                    f"Page {page.page_number} has no original image."
                )

    @staticmethod
    def cleanup(
        document: Document,
    ) -> None:
        """
        Delete temporary files created during processing.

        Original uploaded files are never deleted.
        """

        try:

            for page in document.pages:

                assets = page.assets

                if assets is None:
                    continue

                for field in fields(assets):
                    
                    value = getattr(
                        assets,
                        field.name,
                    )

                    if not isinstance(value, Path):
                        continue

                    if (
                        value.exists()
                        and value != assets.original
                    ):

                        try:

                            value.unlink()

                            app_logger.debug(
                                f"Deleted temporary file: {value}"
                            )

                        except Exception as exc:

                            app_logger.warning(
                                f"Failed to delete {value}: {exc}"
                            )

        except Exception as exc:

            app_logger.warning(
                f"Cleanup failed: {exc}"
            )


document_pipeline = DocumentPipeline()