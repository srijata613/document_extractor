from __future__ import annotations

from typing import List

import cv2
import torch
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from models.document import Document, OCRWord, Page
from services.model_manager import model_manager
from utils.image_utils import image_utils
from utils.logger import app_logger


class HandwritingRecognitionError(Exception):
    """Raised when handwriting recognition fails."""


class HandwritingService:
    """
    Handwritten text recognition using Microsoft's TrOCR.
    """

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

            crops = self._extract_candidate_regions(
                image
            )

            for crop, bbox in crops:

                text = self._recognize_crop(
                    crop
                )

                if not text:
                    continue

                page.add_handwriting_word(
                    OCRWord(
                        text=text,
                        confidence=1.0,
                        bbox=bbox,
                        source="handwritten",
                    )
                )

            app_logger.info(
                f"Page {page.page_number}: "
                f"{len(page.handwriting_words)} handwritten words."
            )

        except Exception as exc:

            app_logger.exception(exc)

            raise HandwritingRecognitionError(
                f"Handwriting recognition failed "
                f"for page {page.page_number}"
            ) from exc


    def _recognize_crop(
        self,
        crop,
    ) -> str:

        pil = image_utils.cv_to_pil(crop)

        pixel_values = self.processor(
            images=pil,
            return_tensors="pt",
        ).pixel_values

        pixel_values = pixel_values.to(
            self.device
        )

        with torch.no_grad():

            generated_ids = (
                self.model.generate(
                    pixel_values
                )
            )

        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        return text.strip()


    def _extract_candidate_regions(
        self,
        image,
    ) -> List[tuple]:

        gray = image_utils.to_gray(image)

        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25,
            12,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 3),
        )

        thresh = cv2.dilate(
            thresh,
            kernel,
            iterations=1,
        )

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates = []

        for contour in contours:

            x, y, w, h = cv2.boundingRect(
                contour
            )

            if w < 25 or h < 12:
                continue

            if w > image.shape[1] * 0.95:
                continue

            crop = image[
                y : y + h,
                x : x + w,
            ]

            bbox = [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h],
            ]

            candidates.append(
                (
                    crop,
                    bbox,
                )
            )

        candidates.sort(
            key=lambda c: (
                c[1][0][1],
                c[1][0][0],
            )
        )

        return candidates
    

    @staticmethod
    def remove_duplicates(
        page: Page,
        iou_threshold: float = 0.50,
    ) -> None:
        """
        Remove duplicate handwritten detections using IoU.
        """

        if len(page.handwriting_words) <= 1:
            return

        filtered: list[OCRWord] = []

        for word in sorted(
            page.handwriting_words,
            key=lambda w: w.confidence,
            reverse=True,
        ):

            duplicate = False

            for existing in filtered:

                if (
                    HandwritingService._iou(
                        word.bbox,
                        existing.bbox,
                    )
                    >= iou_threshold
                ):
                    duplicate = True
                    break

            if not duplicate:
                filtered.append(word)

        page.handwriting_words = filtered


    @staticmethod
    def estimate_confidence(
        text: str,
    ) -> float:
        """
        Simple confidence estimation for TrOCR output.
        """

        if not text:
            return 0.0

        score = 1.0

        if len(text) < 2:
            score -= 0.30

        if text.count("?") > 0:
            score -= 0.20

        if any(ch in "@#$%^&*" for ch in text):
            score -= 0.20

        return max(score, 0.10)


    def postprocess_page(
        self,
        page: Page,
    ) -> None:
        """
        Clean handwriting results.
        """

        cleaned: list[OCRWord] = []

        for word in page.handwriting_words:

            word.text = " ".join(
                word.text.split()
            ).strip()

            word.confidence = self.estimate_confidence(
                word.text
            )

            if word.text:
                cleaned.append(word)

        page.handwriting_words = cleaned

        self.remove_duplicates(page)


    def process_and_clean(
        self,
        page: Page,
    ) -> None:
        """
        Complete handwriting pipeline.
        """

        self.process_page(page)

        self.postprocess_page(page)


    @staticmethod
    def export_words(
        page: Page,
    ) -> list[dict]:
        """
        Export handwriting results.
        """

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
        Intersection over Union between two OCR boxes.
        """

        x1 = max(
            bbox1[0][0],
            bbox2[0][0],
        )

        y1 = max(
            bbox1[0][1],
            bbox2[0][1],
        )

        x2 = min(
            bbox1[2][0],
            bbox2[2][0],
        )

        y2 = min(
            bbox1[2][1],
            bbox2[2][1],
        )

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