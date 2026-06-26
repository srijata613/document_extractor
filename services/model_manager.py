from __future__ import annotations

from threading import Lock
from typing import Optional

import torch
from paddleocr import PaddleOCR
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from config import HANDWRITING_MODEL, OCR_LANGUAGE, USE_GPU
from utils.logger import app_logger


class ModelInitializationError(RuntimeError):
    """Raised when a model cannot be initialized."""


class ModelManager:
    """
    Thread-safe singleton responsible for managing all AI models.

    Models are loaded lazily and reused across the application.
    """

    _instance: Optional["ModelManager"] = None
    _instance_lock = Lock()

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()

        return cls._instance

    def _initialize(self) -> None:
        self._ocr_lock = Lock()
        self._trocr_lock = Lock()

        self._ocr: Optional[PaddleOCR] = None
        self._trocr_processor: Optional[TrOCRProcessor] = None
        self._trocr_model: Optional[VisionEncoderDecoderModel] = None

    # PaddleOCR

    def get_ocr(self) -> PaddleOCR:
        """
        Returns the shared PaddleOCR instance.
        """

        if self._ocr is None:
            with self._ocr_lock:
                if self._ocr is None:
                    try:
                        app_logger.info("Loading PaddleOCR...")

                        self._ocr = PaddleOCR(
                            lang=OCR_LANGUAGE,
                            use_gpu=USE_GPU,
                            use_doc_orientation_classify=False,
                            use_doc_unwarping=False,
                            use_textline_orientation=False,
                        )

                        app_logger.success("PaddleOCR loaded successfully.")

                    except Exception as exc:
                        app_logger.exception(exc)
                        raise ModelInitializationError(
                            "Failed to initialize PaddleOCR."
                        ) from exc

        return self._ocr

    # TrOCR

    def get_trocr_processor(self) -> TrOCRProcessor:
        """
        Returns the shared TrOCR processor.
        """

        if self._trocr_processor is None:
            with self._trocr_lock:
                if self._trocr_processor is None:
                    try:
                        app_logger.info("Loading TrOCR processor...")

                        self._trocr_processor = (
                            TrOCRProcessor.from_pretrained(
                                HANDWRITING_MODEL
                            )
                        )

                        app_logger.success("TrOCR processor loaded.")

                    except Exception as exc:
                        app_logger.exception(exc)
                        raise ModelInitializationError(
                            "Failed to initialize TrOCR processor."
                        ) from exc

        return self._trocr_processor

    def get_trocr_model(self) -> VisionEncoderDecoderModel:
        """
        Returns the shared TrOCR model.
        """

        if self._trocr_model is None:
            with self._trocr_lock:
                if self._trocr_model is None:
                    try:
                        app_logger.info("Loading TrOCR model...")

                        self._trocr_model = (
                            VisionEncoderDecoderModel.from_pretrained(
                                HANDWRITING_MODEL
                            )
                        )

                        self._trocr_model.eval()

                        if USE_GPU and torch.cuda.is_available():
                            self._trocr_model.to("cuda")
                            app_logger.success(
                                "TrOCR initialized on CUDA."
                            )
                        else:
                            app_logger.success(
                                "TrOCR initialized on CPU."
                            )

                    except Exception as exc:
                        app_logger.exception(exc)
                        raise ModelInitializationError(
                            "Failed to initialize TrOCR model."
                        ) from exc

        return self._trocr_model

    # Utility

    def preload_models(self) -> None:
        """
        Load every model into memory.
        """

        app_logger.info("Preloading AI models...")

        self.get_ocr()
        self.get_trocr_processor()
        self.get_trocr_model()

        app_logger.success("All AI models loaded.")

    def is_ready(self) -> bool:
        """
        Returns True if every model has been initialized.
        """

        return all(
            (
                self._ocr is not None,
                self._trocr_processor is not None,
                self._trocr_model is not None,
            )
        )


model_manager = ModelManager()