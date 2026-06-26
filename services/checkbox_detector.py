from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from pydoc import text

import cv2
import numpy as np

from models.document import (
    Checkbox,
    Document,
    OCRWord,
    Page,
)
from utils.image_utils import image_utils
from utils.logger import app_logger


class CheckboxDetectionError(Exception):
    """Raised when checkbox detection fails."""


@dataclass(slots=True)
class CheckboxCandidate:
    """
    Internal representation of a possible checkbox.
    """

    contour: np.ndarray

    bbox: tuple[int, int, int, int]

    contour_area: float

    perimeter: float

    aspect_ratio: float

    fill_ratio: float = 0.0

    border_complete: bool = False

    connected_components: int = 0

    mark_type: str = "unknown"

    confidence: float = 0.0


class CheckboxDetector:

    """
    Production checkbox detector.

    Supports

    ✓ Tick

    ✗ Cross

    ■ Filled

    Empty

    Scribble

    Rotated forms

    Low quality scans
    """

    MIN_SIZE = 12

    MAX_SIZE = 65

    MIN_AREA = 90

    MAX_AREA = 4500

    MIN_ASPECT = 0.75

    MAX_ASPECT = 1.30

    def process_document(
        self,
        document: Document,
    ) -> Document:

        app_logger.info(
            "Detecting checkboxes..."
        )

        for page in document.pages:

            self.process_page(page)

        app_logger.success(
            "Checkbox detection completed."
        )

        return document

    def process_page(
        self,
        page: Page,
    ) -> None:

        image_path = (
            page.assets.checkbox_image
            or page.assets.threshold
            or page.assets.preprocessed
            or page.assets.original
        )

        try:

            image = image_utils.load_cv_image(
                image_path
            )

            binary = self._prepare_binary(
                image
            )

            candidates = self._find_candidates(
                binary
            )

            for candidate in candidates:

                checkbox = self._classify_candidate(
                    candidate,
                    binary,
                )

                if checkbox:

                    page.add_checkbox(
                        checkbox
                    )

            app_logger.info(
                f"Detected "
                f"{len(page.checkboxes)} "
                f"checkboxes on page "
                f"{page.page_number}"
            )

        except Exception as exc:

            app_logger.exception(
                exc
            )

            raise CheckboxDetectionError(
                f"Checkbox detection failed "
                f"for page "
                f"{page.page_number}"
            ) from exc

    @staticmethod
    def _prepare_binary(
        image: np.ndarray,
    ) -> np.ndarray:

        gray = image_utils.to_gray(
            image
        )

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0,
        )

        binary = cv2.adaptiveThreshold(

            gray,

            255,

            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

            cv2.THRESH_BINARY_INV,

            31,

            12,
        )

        kernel = cv2.getStructuringElement(

            cv2.MORPH_RECT,

            (2, 2),
        )

        binary = cv2.morphologyEx(

            binary,

            cv2.MORPH_CLOSE,

            kernel,

            iterations=1,
        )

        return binary

    def _find_candidates(
        self,
        binary: np.ndarray,
    ) -> list[CheckboxCandidate]:

        contours, _ = cv2.findContours(

            binary,

            cv2.RETR_LIST,

            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates = []

        for contour in contours:

            candidate = self._build_candidate(

                contour
            )

            if candidate is None:
                continue

            candidates.append(
                candidate
            )

        return candidates

    def _build_candidate(
        self,
        contour,
    ) -> CheckboxCandidate | None:

        area = cv2.contourArea(
            contour
        )

        if area < self.MIN_AREA:
            return None

        if area > self.MAX_AREA:
            return None

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if (
            w < self.MIN_SIZE
            or h < self.MIN_SIZE
        ):
            return None

        if (
            w > self.MAX_SIZE
            or h > self.MAX_SIZE
        ):
            return None

        aspect = w / float(h)

        if not (
            self.MIN_ASPECT
            <= aspect
            <= self.MAX_ASPECT
        ):
            return None

        return CheckboxCandidate(

            contour=contour,

            bbox=(x, y, w, h),

            contour_area=area,

            perimeter=perimeter,

            aspect_ratio=aspect,
        )
        
    def _classify_candidate(
        self,
        candidate: CheckboxCandidate,
        binary: np.ndarray,
    ) -> Checkbox | None:

        x, y, w, h = candidate.bbox

        roi = binary[y:y + h, x:x + w]

        if roi.size == 0:
            return None

        border = max(2, int(min(w, h) * 0.10))

        if (
            roi.shape[0] <= border * 2
            or roi.shape[1] <= border * 2
        ):
            return None

        inner = roi[
            border:-border,
            border:-border,
        ]

        candidate.fill_ratio = self._calculate_fill_ratio(
            inner
        )

        candidate.connected_components = (
            self._count_connected_components(
                inner
            )
        )

        candidate.border_complete = (
            self._is_border_complete(
                roi
            )
        )

        candidate.mark_type = self._detect_mark_type(
            inner,
            candidate.fill_ratio,
            candidate.connected_components,
        )

        candidate.confidence = self._calculate_confidence(
            candidate
        )

        return Checkbox(
            selected=candidate.mark_type != "empty",
            checked=candidate.mark_type != "empty",
            confidence=candidate.confidence,
            bbox=[x, y, w, h],
            mark_type=candidate.mark_type,
            fill_ratio=candidate.fill_ratio,
            area=candidate.contour_area,
            aspect_ratio=candidate.aspect_ratio,
            perimeter=candidate.perimeter,
            connected_components=candidate.connected_components,
            border_complete=candidate.border_complete,
        )

    @staticmethod
    def _calculate_fill_ratio(
        roi: np.ndarray,
    ) -> float:

        total_pixels = roi.shape[0] * roi.shape[1]

        if total_pixels == 0:
            return 0.0

        filled = cv2.countNonZero(
            roi
        )

        return filled / float(total_pixels)

    @staticmethod
    def _count_connected_components(
        roi: np.ndarray,
    ) -> int:

        count, _ = cv2.connectedComponents(
            roi
        )

        return max(0, count - 1)

    @staticmethod
    def _is_border_complete(
        roi: np.ndarray,
    ) -> bool:

        height, width = roi.shape[:2]

        border = 2

        top = roi[:border, :]
        bottom = roi[height - border:, :]
        left = roi[:, :border]
        right = roi[:, width - border:]

        pixels = (
            cv2.countNonZero(top)
            + cv2.countNonZero(bottom)
            + cv2.countNonZero(left)
            + cv2.countNonZero(right)
        )

        total = (
            top.size
            + bottom.size
            + left.size
            + right.size
        )

        if total == 0:
            return False

        return pixels / total > 0.60

    def _detect_mark_type(
        self,
        roi: np.ndarray,
        fill_ratio: float,
        components: int,
    ) -> str:

        if fill_ratio < 0.03:
            return "empty"

        if fill_ratio > 0.75:
            return "filled"

        lines = cv2.HoughLinesP(
            roi,
            1,
            np.pi / 180,
            threshold=12,
            minLineLength=5,
            maxLineGap=3,
        )

        if lines is not None:

            angles = []

            for line in lines:

                x1, y1, x2, y2 = line[0]

                angle = abs(
                    np.degrees(
                        np.arctan2(
                            y2 - y1,
                            x2 - x1,
                        )
                    )
                )

                angles.append(angle)

            positive = any(
                20 <= a <= 70
                for a in angles
            )

            negative = any(
                110 <= a <= 160
                for a in angles
            )

            if positive and negative:
                return "cross"

            if positive:
                return "tick"

        if (
            components > 6
            and fill_ratio > 0.20
        ):
            return "scribble"

        return "unknown"

    @staticmethod
    def _calculate_confidence(
        candidate: CheckboxCandidate,
    ) -> float:

        score = 0.40

        score += min(
            candidate.fill_ratio,
            0.40,
        )

        if candidate.border_complete:
            score += 0.20

        if (
            0.90
            <= candidate.aspect_ratio
            <= 1.10
        ):
            score += 0.15

        if candidate.mark_type != "unknown":
            score += 0.15

        if candidate.connected_components <= 4:
            score += 0.10

        return min(score, 1.0)
    

    def assign_labels(
        self,
        page: Page,
        max_distance: int = 220,
        line_tolerance: int = 18,
    ) -> None:
        """
        Associate each checkbox with the nearest OCR text.
        """

        if not page.ocr_words:
            return
        
        page.ocr_words.sort(
            key=lambda w: (
                min(p[1] for p in w.bbox),
                min(p[0] for p in w.bbox),
            )
        )

        for checkbox in page.checkboxes:

            x, y, w, h = checkbox.bbox

            center_y = y + h // 2
            
            words = []

            for word in page.ocr_words:

                left = min(p[0] for p in word.bbox)
                right = max(p[0] for p in word.bbox)
                
                top = min(p[1] for p in word.bbox)
                bottom = max(p[1] for p in word.bbox)
                
                word_center_y = (top + bottom) // 2

                # Ignore labels on previous lines
                if abs(word_center_y - center_y) > line_tolerance:
                    continue

                # Labels are usually to the right
                if left < x:
                    continue
                
                if left - (x + w) > max_distance:
                    continue
                
                words.append(
                    (
                        left,
                        word.text.strip(),
                    )
                )
                
            if not words:
                continue
            
            words.sort(key=lambda item: item[0])
            
            lable = " ".join(
                text
                for _, text in words
            )
            
            checkbox.label = lable
            checkbox.label = lable

    def infer_groups(
        self,
        page: Page,
    ) -> None:
        """
        Infer logical checkbox groups.
        """

        groups = {

            "Gender": {
                "male",
                "female",
                "other",
            },

            "Category": {
                "sc",
                "st",
                "obc",
                "general",
                "ews",
            },

            "Marital Status": {
                "married",
                "unmarried",
                "widow",
                "divorced",
            },

            "Yes/No": {
                "yes",
                "no",
            },

            "Residential": {
                "rural",
                "urban",
            },

            "Disability": {
                "yes",
                "no",
            },

        }

        for checkbox in page.checkboxes:

            label = checkbox.label.lower()

            for group, values in groups.items():

                if label in values:

                    checkbox.group = group

                    checkbox.value = checkbox.label

                    break


    @staticmethod
    def remove_duplicates(
        page: Page,
        overlap: float = 0.55,
    ) -> None:

        filtered = []

        ordered = sorted(

            page.checkboxes,

            key=lambda c: c.confidence,

            reverse=True,
        )

        for candidate in ordered:

            duplicate = False

            for existing in filtered:

                if (

                    CheckboxDetector._iou(

                        candidate.bbox,

                        existing.bbox,

                    )

                    >= overlap

                ):

                    duplicate = True

                    break

            if not duplicate:

                filtered.append(
                    candidate
                )

        page.checkboxes = filtered


    def process_and_clean(
        self,
        page: Page,
    ) -> None:

        self.process_page(
            page
        )

        self.remove_duplicates(
            page
        )

        self.assign_labels(
            page
        )

        self.infer_groups(
            page
        )


    @staticmethod
    def export_checkboxes(
        page: Page,
    ) -> list[dict]:

        return [

            {

                "label": cb.label,

                "group": cb.group,

                "value": cb.value,

                "selected": cb.selected,

                "checked": cb.checked,

                "mark_type": cb.mark_type,

                "confidence": cb.confidence,

                "bbox": cb.bbox,

            }

            for cb in page.checkboxes

        ]


    @staticmethod
    def _iou(
        box1,
        box2,
    ) -> float:

        x1 = max(
            box1[0],
            box2[0],
        )

        y1 = max(
            box1[1],
            box2[1],
        )

        x2 = min(
            box1[0] + box1[2],
            box2[0] + box2[2],
        )

        y2 = min(
            box1[1] + box1[3],
            box2[1] + box2[3],
        )

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (
            (x2 - x1)
            * (y2 - y1)
        )

        area1 = (
            box1[2]
            * box1[3]
        )

        area2 = (
            box2[2]
            * box2[3]
        )

        union = (
            area1
            + area2
            - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union


checkbox_detector = CheckboxDetector()