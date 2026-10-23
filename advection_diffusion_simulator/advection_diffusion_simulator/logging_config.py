"""Logging configuration utilities for SWE Simulator."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging for SWE Simulator.

    Parameters
    ----------
    level : int or str, default=logging.INFO
        Logging level (e.g., logging.DEBUG, "INFO", "WARNING")
    log_file : str or Path, optional
        If provided, logs will also be written to this file
    format_string : str, optional
        Custom format string for log messages

    Returns
    -------
    logging.Logger
        Configured logger instance for 'swe_simulator'
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger("swe_simulator")
    logger.setLevel(level)
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Parameters
    ----------
    name : str
        Name for the logger (typically __name__)

    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level(level: Union[int, str]) -> None:
    """
    Change the logging level for all SWE Simulator loggers.

    Parameters
    ----------
    level : int or str
        New logging level
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    logger = logging.getLogger("swe_simulator")
    logger.setLevel(level)

    for handler in logger.handlers:
        handler.setLevel(level)


def disable_logging() -> None:
    """Disable all logging output from SWE Simulator."""
    logger = logging.getLogger("swe_simulator")
    logger.disabled = True


def enable_logging() -> None:
    """Re-enable logging output after it has been disabled."""
    logger = logging.getLogger("swe_simulator")
    logger.disabled = False
