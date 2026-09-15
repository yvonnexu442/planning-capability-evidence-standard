"""Logging helpers for research scripts."""

import logging
from pathlib import Path
from typing import Optional, Union


def setup_logger(
    name: str,
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a logger with console output and optional file output.

    Logging matters for reproducible research because it records which pipeline
    steps ran and where outputs were written. It is preferred over ad hoc print
    statements in scripts.

    The function avoids adding duplicate handlers when called repeatedly.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_path) for handler in logger.handlers):
            file_handler = logging.FileHandler(str(log_path))
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)

    return logger
