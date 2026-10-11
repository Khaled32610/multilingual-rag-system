"""
Centralized logging setup for rag_system.

Provides a single `get_logger(name)` function that returns a logger
writing to both the console and a rotating log file, configured from
Settings (configs/config.yaml + .env).

Usage:
    from rag_system.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rag_system.utils.config import get_settings

_CONFIGURED_LOGGERS: set[str] = set()

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _build_handlers(log_level: str, logs_dir: str, file_name: str,
                     max_bytes: int, backup_count: int) -> list[logging.Handler]:
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_file_path = Path(logs_dir) / file_name

    file_handler = RotatingFileHandler(
        filename=str(log_file_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    return [console_handler, file_handler]


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Configuration (level, log file location, rotation policy) is pulled
    from the global Settings object. Handlers are only attached once per
    logger name, so repeated calls are safe and cheap.
    """
    logger = logging.getLogger(name)

    if name in _CONFIGURED_LOGGERS:
        return logger

    settings = get_settings()
    log_level = settings.logging.level.upper()

    logger.setLevel(log_level)
    logger.propagate = False

    for handler in _build_handlers(
        log_level=log_level,
        logs_dir=settings.paths.logs_dir,
        file_name=settings.logging.file_name,
        max_bytes=settings.logging.max_bytes,
        backup_count=settings.logging.backup_count,
    ):
        logger.addHandler(handler)

    _CONFIGURED_LOGGERS.add(name)
    return logger