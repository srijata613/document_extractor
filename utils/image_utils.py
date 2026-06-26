from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from utils.logger import app_logger


class ImageUtils:
    """
    Utility functions for image loading, conversion and manipulation.
    """

    @staticmethod
    def load_cv_image(image_path: Path) -> np.ndarray:
        """
        Load image using OpenCV.
        """
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(f"Unable to load image: {image_path}")

        return image

    @staticmethod
    def save_cv_image(image: np.ndarray, output_path: Path) -> None:
        """
        Save OpenCV image.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(output_path), image):
            raise IOError(f"Failed to save image: {output_path}")

    @staticmethod
    def cv_to_pil(image: np.ndarray) -> Image.Image:
        """
        Convert OpenCV image to PIL.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    @staticmethod
    def pil_to_cv(image: Image.Image) -> np.ndarray:
        """
        Convert PIL image to OpenCV.
        """
        rgb = np.array(image)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def to_gray(image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale.
        """
        if len(image.shape) == 2:
            return image

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def resize(
        image: np.ndarray,
        width: int | None = None,
        height: int | None = None,
    ) -> np.ndarray:
        """
        Resize while preserving aspect ratio.
        """

        if width is None and height is None:
            return image

        h, w = image.shape[:2]

        if width is None:
            ratio = height / h
            width = int(w * ratio)

        elif height is None:
            ratio = width / w
            height = int(h * ratio)

        return cv2.resize(
            image,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def rotate(
        image: np.ndarray,
        angle: float,
    ) -> np.ndarray:
        """
        Rotate image.
        """

        h, w = image.shape[:2]

        center = (w // 2, h // 2)

        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        return cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    @staticmethod
    def crop(
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Crop image.
        """

        return image[
            y:y + height,
            x:x + width,
        ]

    @staticmethod
    def draw_bbox(
        image: np.ndarray,
        bbox: list[list[int]],
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw OCR bounding box.
        """

        pts = np.array(bbox, dtype=np.int32)

        cv2.polylines(
            image,
            [pts],
            True,
            color,
            thickness,
        )

        return image

    @staticmethod
    def draw_rectangle(
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        color: tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw rectangle.
        """

        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            color,
            thickness,
        )

        return image

    @staticmethod
    def image_size(image: np.ndarray) -> tuple[int, int]:
        """
        Returns width, height.
        """

        h, w = image.shape[:2]

        return w, h

    @staticmethod
    def exists(image_path: Path) -> bool:
        return image_path.exists()

    @staticmethod
    def clone(image: np.ndarray) -> np.ndarray:
        return image.copy()

    @staticmethod
    def log_image_info(
        image: np.ndarray,
        image_name: str,
    ) -> None:
        """
        Log image dimensions.
        """

        h, w = image.shape[:2]

        app_logger.debug(
            f"{image_name}: {w}x{h}"
        )


image_utils = ImageUtils()