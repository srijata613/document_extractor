from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse

from config.settings import settings
from services.pipeline import (
    PipelineError,
    document_pipeline,
)
from utils.logger import app_logger
from uuid import uuid4


router = APIRouter(
    prefix="/upload",
    tags=["Document Extraction"],
)


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}


MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

def _validate_extension(
    filename: str,
) -> None:

    extension = Path(
        filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type."
            ),
        )


async def _validate_size(
    file: UploadFile,
) -> None:

    size = 0

    while chunk := await file.read(
        1024 * 1024
    ):

        size += len(chunk)

        if size > MAX_FILE_SIZE:

            raise HTTPException(

                status_code=413,

                detail=(
                    "Uploaded file "
                    "is too large."
                ),
            )

    await file.seek(0)


@router.post(
    "/",
    response_class=FileResponse,
)
async def upload_document(
    file: UploadFile = File(...),
):

    if file.filename is None:

        raise HTTPException(
            status_code=400,
            detail="No filename supplied.",
        )

    _validate_extension(
        file.filename,
    )

    await _validate_size(
        file,
    )

    uploads = Path(
        settings.UPLOAD_DIRECTORY
    )

    outputs = Path(
        settings.OUTPUT_DIRECTORY
    )

    uploads.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_filename = Path(file.filename)
    
    unique_filename = (
        f"{uuid4().hex}"
        f"{original_filename.suffix.lower()}"
    )

    input_path = uploads / unique_filename

    with input_path.open(
        "wb",
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer,
        )

    app_logger.info(
        f"Uploaded {file.filename}"
    )

    try:

        excel = document_pipeline.process(
            input_path,
            outputs,
            output_filename=original_filename.stem,
        )

        download_name = (
            f"{original_filename.stem}.xlsx"
        )

    except PipelineError as exc:

        raise HTTPException(

            status_code=500,

            detail=str(exc),

        ) from exc

    return FileResponse(
        excel,
        filename=download_name,
        media_type=(
            "application/"
            "vnd.openxmlformats-"
            "officedocument."
            "spreadsheetml.sheet"
        ),
    )