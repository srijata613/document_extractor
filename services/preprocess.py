from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config import (
    ENABLE_CONTRAST,
    ENABLE_DENOISE,
    ENABLE_DESKEW,
    PREPROCESSED_DIR,
    THRESHOLD_DIR,
)
from models.document import Document, Page
from utils.image_utils import image_utils
from utils.logger import app_logger


class PreprocessingError(Exception):
    """Raised when image preprocessing fails."""


class PreprocessService:
    """
    Image preprocessing pipeline.

    Pipeline:
        Load
        ↓
        Deskew
        ↓
        Denoise
        ↓
        Contrast Enhancement
        ↓
        Adaptive Threshold
        ↓
        Morphological Cleanup
        ↓
        Save
    """

    def process_document(
        self,
        document: Document,
    ) -> Document:
        """
        Standard pipeline entry point.
        Keeps the interface consistent with the other services.
        """
        return self.preprocess_document(document)

    def preprocess_document(
        self,
        document: Document,
    ) -> Document:

        app_logger.info(
            f"Preprocessing {document.total_pages} pages..."
        )

        for page in document.pages:
            self.preprocess_page(page)

        app_logger.success(
            "Preprocessing completed."
        )

        return document

    def preprocess_page(self, page: Page) -> None:

        try:

            image = image_utils.load_cv_image(
                page.assets.original
            )

            if ENABLE_DESKEW:
                image = self._deskew(image)

            if ENABLE_DENOISE:
                image = self._denoise(image)

            if ENABLE_CONTRAST:
                image = self._enhance_contrast(image)

            threshold = self._adaptive_threshold(image)

            threshold = self._morphological_cleanup(
                threshold
            )

            processed_path = (
                PREPROCESSED_DIR
                / page.assets.original.name
            )

            threshold_path = (
                THRESHOLD_DIR
                / page.assets.original.name
            )

            image_utils.save_cv_image(
                image,
                processed_path,
            )

            image_utils.save_cv_image(
                threshold,
                threshold_path,
            )

            page.assets.preprocessed = processed_path
            page.assets.threshold = threshold_path

            page.assets.ocr_image = threshold_path
            page.assets.handwriting_image = processed_path
            page.assets.checkbox_image = threshold_path

        except Exception as exc:

            app_logger.exception(exc)

            raise PreprocessingError(
                f"Failed preprocessing page {page.page_number}"
            ) from exc

    @staticmethod
    def _deskew(
        image: np.ndarray,
    ) -> np.ndarray:

        gray = image_utils.to_gray(image)

        gray = cv2.bitwise_not(gray)

        _, thresh = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )

        coords = np.column_stack(
            np.where(thresh > 0)
        )

        if len(coords) == 0:
            return image

        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        return image_utils.rotate(
            image,
            angle,
        )

    @staticmethod
    def _denoise(
        image: np.ndarray,
    ) -> np.ndarray:

        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            10,
            10,
            7,
            21,
        )

    @staticmethod
    def _enhance_contrast(
        image: np.ndarray,
    ) -> np.ndarray:

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB,
        )

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        l = clahe.apply(l)

        merged = cv2.merge((l, a, b))

        return cv2.cvtColor(
            merged,
            cv2.COLOR_LAB2BGR,
        )

    @staticmethod
    def _adaptive_threshold(
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Create a clean binary image for OCR.
        """

        gray = image_utils.to_gray(image)

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0,
        )

        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15,
        )

    @staticmethod
    def _morphological_cleanup(
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Remove isolated noise while preserving characters.
        """

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (2, 2),
        )

        image = cv2.morphologyEx(
            image,
            cv2.MORPH_OPEN,
            kernel,
        )

        image = cv2.morphologyEx(
            image,
            cv2.MORPH_CLOSE,
            kernel,
        )

        return image

    @staticmethod
    def remove_shadows(
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Shadow removal using morphological dilation.
        """

        rgb_planes = cv2.split(image)

        result_planes = []

        for plane in rgb_planes:

            dilated = cv2.dilate(
                plane,
                np.ones((7, 7), np.uint8),
            )

            background = cv2.medianBlur(
                dilated,
                21,
            )

            diff = 255 - cv2.absdiff(
                plane,
                background,
            )

            normalized = cv2.normalize(
                diff,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8UC1,
            )

            result_planes.append(normalized)

        return cv2.merge(result_planes)

    @staticmethod
    def sharpen(
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Sharpen text regions.
        """

        kernel = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ],
            dtype=np.float32,
        )

        return cv2.filter2D(
            image,
            -1,
            kernel,
        )

    @staticmethod
    def resize_for_ocr(
        image: np.ndarray,
        min_width: int = 1800,
    ) -> np.ndarray:
        """
        Upscale small images for better OCR.
        """

        height, width = image.shape[:2]

        if width >= min_width:
            return image

        scale = min_width / width

        return cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    @staticmethod
    def validate_image(
        image: np.ndarray,
    ) -> None:

        if image is None:
            raise PreprocessingError(
                "Image is None."
            )

        if image.size == 0:
            raise PreprocessingError(
                "Empty image."
            )

        if len(image.shape) not in (2, 3):
            raise PreprocessingError(
                "Unsupported image dimensions."
            )

    @staticmethod
    def save_debug_image(
        image: np.ndarray,
        output_path: Path,
    ) -> None:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        image_utils.save_cv_image(
            image,
            output_path,
        )


preprocess_service = PreprocessService()