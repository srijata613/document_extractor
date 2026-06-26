from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
import torch
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from models.document import (
    Document,
    OCRWord,
    Page,
)
from services.model_manager import model_manager
from utils.image_utils import image_utils
from utils.logger import app_logger


class HandwritingRecognitionError(Exception):
    """Raised when handwriting recognition fails."""


@dataclass(slots=True)
class HandwritingRegion:
    """
    Candidate handwriting region before OCR.
    """

    image: np.ndarray
    bbox: list[list[int]]

    ocr_confidence: float

    trocr_confidence: float = 0.0

    text: str = ""


class HandwritingService:
    """
    Production handwriting recognition service.

    Pipeline

        OCR Results
              ↓
        Candidate Detection
              ↓
        Region Merge
              ↓
        TrOCR Recognition
              ↓
        Confidence Fusion
              ↓
        Duplicate Removal
              ↓
        OCRWord Export
    """

    # Ignore tiny regions

    MIN_WIDTH = 45
    MIN_HEIGHT = 18

    # Ignore regions larger than page

    MAX_WIDTH_RATIO = 0.90

    # OCR confidence below this
    # becomes handwriting candidate

    LOW_CONFIDENCE = 0.70

    # Merge nearby regions

    MERGE_X_GAP = 30
    MERGE_Y_GAP = 18

    # Batch size for TrOCR

    BATCH_SIZE = 8

    def __init__(self) -> None:

        self.processor: TrOCRProcessor = (
            model_manager.get_trocr_processor()
        )

        self.model: VisionEncoderDecoderModel = (
            model_manager.get_trocr_model()
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        if (
            next(self.model.parameters()).device.type
            != self.device
        ):
            self.model.to(self.device)

    def process_document(
        self,
        document: Document,
    ) -> Document:

        app_logger.info(
            "Starting handwriting recognition..."
        )

        for page in document.pages:

            self.process_page(page)

        app_logger.success(
            "Handwriting recognition completed."
        )

        return document

    def process_page(
        self,
        page: Page,
    ) -> None:

        image_path = (
            page.assets.handwriting_image
            or page.assets.preprocessed
            or page.assets.original
        )

        try:

            image = image_utils.load_cv_image(
                image_path
            )

            regions = self._collect_candidate_regions(
                page,
                image,
            )

            if not regions:

                app_logger.info(
                    f"Page {page.page_number}: "
                    "no handwriting candidates."
                )

                return

            regions = self._merge_regions(
                regions,
            )

            self._recognize_regions(
                page,
                image,
                regions,
            )

            app_logger.info(
                f"Page {page.page_number}: "
                f"{len(page.handwriting_words)} "
                "handwriting regions recognized."
            )

        except Exception as exc:

            app_logger.exception(exc)

            raise HandwritingRecognitionError(
                f"Handwriting recognition failed "
                f"for page {page.page_number}"
            ) from exc


    def _collect_candidate_regions(
        self,
        page: Page,
        image: np.ndarray,
    ) -> list[HandwritingRegion]:

        candidates: list[
            HandwritingRegion
        ] = []

        if not page.ocr_words:
            return candidates

        for word in page.ocr_words:

            if word.confidence >= self.LOW_CONFIDENCE:
                continue

            x1 = int(min(p[0] for p in word.bbox))
            y1 = int(min(p[1] for p in word.bbox))

            x2 = int(max(p[0] for p in word.bbox))
            y2 = int(max(p[1] for p in word.bbox))

            width = x2 - x1
            height = y2 - y1

            if width < self.MIN_WIDTH:
                continue

            if height < self.MIN_HEIGHT:
                continue

            if width > (
                image.shape[1]
                * self.MAX_WIDTH_RATIO
            ):
                continue

            pad = 8

            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)

            x2 = min(
                image.shape[1],
                x2 + pad,
            )

            y2 = min(
                image.shape[0],
                y2 + pad,
            )

            crop = image[
                y1:y2,
                x1:x2,
            ]

            candidates.append(
                HandwritingRegion(
                    image=np.empty((0, 0), dtype=np.uint8),
                    bbox=[
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2],
                    ],
                    ocr_confidence=word.confidence,
                )
            )

        return candidates


    def _merge_regions(
        self,
        regions: list[
            HandwritingRegion
        ],
    ) -> list[
        HandwritingRegion
    ]:

        if len(regions) <= 1:
            return regions

        regions.sort(
            key=lambda r: (
                r.bbox[0][1],
                r.bbox[0][0],
            )
        )

        merged = [
            regions[0]
        ]

        for region in regions[1:]:

            previous = merged[-1]

            prev_right = previous.bbox[1][0]
            curr_left = region.bbox[0][0]

            prev_y = previous.bbox[0][1]
            curr_y = region.bbox[0][1]

            if (
                abs(prev_y - curr_y)
                <= self.MERGE_Y_GAP
                and
                curr_left - prev_right
                <= self.MERGE_X_GAP
            ):

                x1 = min(
                    previous.bbox[0][0],
                    region.bbox[0][0],
                )

                y1 = min(
                    previous.bbox[0][1],
                    region.bbox[0][1],
                )

                x2 = max(
                    previous.bbox[2][0],
                    region.bbox[2][0],
                )

                y2 = max(
                    previous.bbox[2][1],
                    region.bbox[2][1],
                )

                previous.bbox = [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2],
                ]

                previous.ocr_confidence = min(
                    previous.ocr_confidence,
                    region.ocr_confidence,
                )

            else:

                merged.append(
                    region
                )

        return merged

    def _recognize_regions(
        self,
        page: Page,
        image: np.ndarray,
        regions: list[HandwritingRegion],
    ) -> None:

        if not regions:
            return

        self._recrop_regions(
            image,
            regions,
        )

        for start in range(
            0,
            len(regions),
            self.BATCH_SIZE,
        ):

            batch = regions[
                start:start + self.BATCH_SIZE
            ]

            images = [
                image_utils.cv_to_pil(
                    region.image
                )
                for region in batch
            ]

            pixel_values = self.processor(
                images=images,
                return_tensors="pt",
                padding=True,
            ).pixel_values.to(
                self.device
            )

            with torch.no_grad():

                generated = self.model.generate(
                    pixel_values,
                    max_new_tokens=64,
                )

            texts = self.processor.batch_decode(
                generated,
                skip_special_tokens=True,
            )

            for region, text in zip(
                batch,
                texts,
            ):

                region.text = text.strip()

                region.trocr_confidence = (
                    self.estimate_confidence(
                        region.text
                    )
                )

                if not region.text:
                    continue

                page.add_handwriting_word(
                    OCRWord(
                        text=region.text,
                        confidence=max(
                            region.trocr_confidence,
                            region.ocr_confidence,
                        ),
                        bbox=region.bbox,
                        source="handwritten",
                    )
                )

    def _recrop_regions(
        self,
        image: np.ndarray,
        regions: list[HandwritingRegion],
    ) -> None:

        height, width = image.shape[:2]

        for region in regions:

            x1 = max(
                0,
                region.bbox[0][0],
            )

            y1 = max(
                0,
                region.bbox[0][1],
            )

            x2 = min(
                width,
                region.bbox[2][0],
            )

            y2 = min(
                height,
                region.bbox[2][1],
            )

            region.image = image[
                y1:y2,
                x1:x2,
            ]

    @staticmethod
    def estimate_confidence(
        text: str,
    ) -> float:
        """
        Estimate confidence from TrOCR output quality.
        """

        if not text:
            return 0.0

        score = 1.0

        length = len(text)

        if length <= 1:
            score -= 0.45
        elif length <= 3:
            score -= 0.15

        invalid = sum(
            c in "@#$%^&*{}[]<>"
            for c in text
        )

        score -= invalid * 0.08

        question = text.count("?")

        score -= question * 0.05

        return max(0.10, min(score, 1.0))


    def postprocess_page(
        self,
        page: Page,
    ) -> None:

        cleaned: list[OCRWord] = []

        for word in page.handwriting_words:

            word.text = (
                " ".join(word.text.split())
                .strip()
            )

            if not word.text:
                continue

            word.confidence = max(
                word.confidence,
                self.estimate_confidence(
                    word.text
                ),
            )

            cleaned.append(word)

        page.handwriting_words = cleaned

        self.remove_duplicates(page)

        page.handwriting_words.sort(
            key=lambda w: (
                min(
                    p[1]
                    for p in w.bbox
                ),
                min(
                    p[0]
                    for p in w.bbox
                ),
            )
        )


    def process_and_clean(
        self,
        page: Page,
    ) -> None:

        self.process_page(page)

        self.postprocess_page(page)


    @staticmethod
    def export_words(
        page: Page,
    ) -> list[dict]:

        return [
            {
                "text": word.text,
                "confidence": word.confidence,
                "bbox": word.bbox,
                "source": word.source,
            }
            for word in page.handwriting_words
        ]
        
    @staticmethod
    def _iou(
        bbox1: list[list[int]],
        bbox2: list[list[int]],
    ) -> float:
        """
        Intersection over Union between two bounding boxes.
        """

        x1 = max(bbox1[0][0], bbox2[0][0])
        y1 = max(bbox1[0][1], bbox2[0][1])

        x2 = min(bbox1[2][0], bbox2[2][0])
        y2 = min(bbox1[2][1], bbox2[2][1])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        area1 = (
            (bbox1[2][0] - bbox1[0][0])
            * (bbox1[2][1] - bbox1[0][1])
        )

        area2 = (
            (bbox2[2][0] - bbox2[0][0])
            * (bbox2[2][1] - bbox2[0][1])
        )

        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union


handwriting_service = HandwritingService()