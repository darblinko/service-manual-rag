"""Logging setup. Library modules only call get_logger; entry points call setup_logging."""

import logging
import sys
from pathlib import Path

from service_manual_rag.config import get_settings

NOISY_LOGGERS = ("httpx", "httpcore", "openai", "urllib3", "neo4j")


def setup_logging(log_level: str | None = None, log_file: Path | None = None) -> None:
    """Configure the root logger from arguments or settings."""
    settings = get_settings()
    level = getattr(logging, (log_level or settings.log_level).upper(), logging.INFO)
    formatter = logging.Formatter(settings.log_format)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_path = log_file or settings.log_file
    if file_path:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""
    return logging.getLogger(name)
