"""
Centralized application logger.

Features:
- Rotating log files
- Colored console logs
- Thread-safe
- Exception backtraces
- Automatic log directory creation
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from config import LOG_FILE, LOG_LEVEL, LOG_RETENTION, LOG_ROTATION

# Remove default logger

logger.remove()

# Console Logger
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    colorize=True,
    backtrace=True,
    diagnose=True,
    enqueue=True,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
)

# File Logger
Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

logger.add(
    LOG_FILE,
    level=LOG_LEVEL,
    rotation=LOG_ROTATION,
    retention=LOG_RETENTION,
    compression="zip",
    enqueue=True,
    backtrace=True,
    diagnose=True,
    encoding="utf-8",
    format=(
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level:<8} | "
        "{name}:{function}:{line} | "
        "{message}"
    ),
)

# Public Logger

app_logger = logger