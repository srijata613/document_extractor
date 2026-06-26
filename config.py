# Central configuration for the Document Extraction System.

from pathlib import Path
from typing import Final

from dotenv import load_dotenv

load_dotenv()

BASE_DIR: Final[Path] = Path(__file__).resolve().parent

UPLOAD_DIR: Final[Path] = BASE_DIR / "uploads"
OUTPUT_DIR: Final[Path] = BASE_DIR / "output"
TEMP_DIR: Final[Path] = BASE_DIR / "temp"
LOG_DIR: Final[Path] = BASE_DIR / "logs"
STATIC_DIR: Final[Path] = BASE_DIR / "static"

RENDERED_DIR: Final[Path] = TEMP_DIR / "rendered"
PREPROCESSED_DIR: Final[Path] = TEMP_DIR / "processed"
THRESHOLD_DIR: Final[Path] = TEMP_DIR / "threshold"
CROPS_DIR: Final[Path] = TEMP_DIR / "crops"

for directory in (
    UPLOAD_DIR,
    OUTPUT_DIR,
    TEMP_DIR,
    LOG_DIR,
    STATIC_DIR,
    RENDERED_DIR,
    PREPROCESSED_DIR,
    THRESHOLD_DIR,
    CROPS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE_MB: Final[int] = 100

ALLOWED_EXTENSIONS: Final[set[str]] = {
    ".pdf",
}

PDF_RENDER_DPI: Final[int] = 300

IMAGE_FORMAT: Final[str] = "png"

ENABLE_DESKEW: Final[bool] = True
ENABLE_DENOISE: Final[bool] = True
ENABLE_CONTRAST: Final[bool] = True

OCR_LANGUAGE: Final[str] = "en"

USE_GPU: Final[bool] = False

OCR_CONFIDENCE_THRESHOLD: Final[float] = 0.45

HANDWRITING_MODEL: Final[str] = "microsoft/trocr-base-handwritten"

CHECKBOX_FILL_THRESHOLD: Final[float] = 0.35

DEFAULT_SHEET_NAME: Final[str] = "Extracted Data"

LOG_FILE: Final[Path] = LOG_DIR / "document_extractor.log"

LOG_ROTATION: Final[str] = "10 MB"

LOG_RETENTION: Final[str] = "30 days"

LOG_LEVEL: Final[str] = "INFO"

MAX_WORKERS: Final[int] = 4

API_TITLE: Final[str] = "Document Extraction API"

API_VERSION: Final[str] = "1.0.0"

API_DESCRIPTION: Final[str] = (
    "Production-ready OCR pipeline for structured document extraction."
)

SUPPORTED_IMAGE_EXTENSIONS: Final[set[str]] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}