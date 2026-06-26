from __future__ import annotations

from pathlib import Path

from paddleocr import PaddleOCR

from models.document import Document, OCRWord, Page
from services.model_manager import model_manager
from utils.logger import app_logger


class OCRProcessingError(Exception):
    """Raised when OCR processing fails."""


class OCRService:
    """
    Printed text OCR using PaddleOCR.
    """

    def __init__(self) -> None:
        self.ocr: PaddleOCR = model_manager.get_ocr()


    def process_document(
        self,
        document: Document,
    ) -> Document:

        app_logger.info(
            "Starting printed OCR..."
        )

        for page in document.pages:
            self.process_page(page)

        app_logger.success(
            "Printed OCR completed."
        )

        return document


    def process_page(
        self,
        page: Page,
    ) -> None:

        image_path = (
            page.assets.ocr_image
            or page.assets.preprocessed
            or page.assets.original
        )

        try:

            result = self.ocr.ocr(
                str(image_path),
                cls=False,
            )

            self._parse_result(
                page,
                result,
            )

        except Exception as exc:

            app_logger.exception(exc)

            raise OCRProcessingError(
                f"OCR failed on page {page.page_number}"
            ) from exc


    def _parse_result(
        self,
        page: Page,
        result,
    ) -> None:

        if not result or not result[0]:
            return
        
        for line in result[0]:
            
            bbox = line[0]
            text = line[1][0]
            score = line[1][1]

            if not text:
                continue

            word = OCRWord(
                    text=text.strip(),
                    confidence=float(score),
                    bbox=bbox,
                    source="printed",
            )

            page.add_ocr_word(
                word
            )

        app_logger.info(
            f"Page {page.page_number}: "
            f"{len(page.ocr_words)} OCR words extracted."
        )
        

    def sort_words(self, page: Page) -> None:
        """
        Sort OCR words from top-to-bottom, left-to-right.
        """

        page.ocr_words.sort(
            key=lambda word: (
                min(point[1] for point in word.bbox),
                min(point[0] for point in word.bbox),
            )
        )


    def filter_low_confidence(
        self,
        page: Page,
        threshold: float = 0.45,
    ) -> None:
        """
        Remove OCR words below confidence threshold.
        """

        page.ocr_words = [
            word
            for word in page.ocr_words
            if word.confidence >= threshold
        ]


    def merge_adjacent_words(
        self,
        page: Page,
        x_gap: int = 18,
        y_gap: int = 12,
    ) -> None:
        """
        Merge fragmented OCR words that belong to the same line.
        """

        if not page.ocr_words:
            return

        self.sort_words(page)

        merged: list[OCRWord] = []

        current = page.ocr_words[0]

        for nxt in page.ocr_words[1:]:

            current_right = max(p[0] for p in current.bbox)
            next_left = min(p[0] for p in nxt.bbox)

            current_y = sum(p[1] for p in current.bbox) / 4
            next_y = sum(p[1] for p in nxt.bbox) / 4

            if (
                abs(current_y - next_y) <= y_gap
                and next_left - current_right <= x_gap
            ):

                current.text += f" {nxt.text}"

                current.confidence = max(
                    current.confidence,
                    nxt.confidence,
                )

                current.bbox = [
                    [
                        min(
                            current.bbox[0][0],
                            nxt.bbox[0][0],
                        ),
                        min(
                            current.bbox[0][1],
                            nxt.bbox[0][1],
                        ),
                    ],
                    [
                        max(
                            current.bbox[1][0],
                            nxt.bbox[1][0],
                        ),
                        min(
                            current.bbox[1][1],
                            nxt.bbox[1][1],
                        ),
                    ],
                    [
                        max(
                            current.bbox[2][0],
                            nxt.bbox[2][0],
                        ),
                        max(
                            current.bbox[2][1],
                            nxt.bbox[2][1],
                        ),
                    ],
                    [
                        min(
                            current.bbox[3][0],
                            nxt.bbox[3][0],
                        ),
                        max(
                            current.bbox[3][1],
                            nxt.bbox[3][1],
                        ),
                    ],
                ]

            else:

                merged.append(current)

                current = nxt

        merged.append(current)

        page.ocr_words = merged


    def get_page_text(
        self,
        page: Page,
    ) -> str:
        """
        Returns page OCR as plain text.
        """

        self.sort_words(page)

        return "\n".join(
            word.text
            for word in page.ocr_words
        )


    def process_and_clean(
        self,
        page: Page,
    ) -> None:
        """
        Complete OCR cleanup pipeline.
        """

        self.process_page(page)

        self.filter_low_confidence(page)

        self.merge_adjacent_words(page)

        self.sort_words(page)


    @staticmethod
    def export_words(
        page: Page,
    ) -> list[dict]:
        """
        Export OCR words for downstream services.
        """

        return [
            {
                "text": word.text,
                "confidence": word.confidence,
                "bbox": word.bbox,
                "source": word.source,
            }
            for word in page.ocr_words
        ]


ocr_service = OCRService()