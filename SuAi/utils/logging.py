import logging
import os
from typing import Optional

_log_level = os.environ.get("LOGLEVEL", logging.getLevelName(logging.WARNING)).upper()
_default_handler: Optional[logging.Handler] = None


def get_logger(name: str) -> logging.Logger:
    global _default_handler

    logger = logging.getLogger(name)
    logger.setLevel(_log_level)

    if _default_handler is None:
        _default_handler = logging.StreamHandler()
        _default_handler.setLevel(_log_level)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        _default_handler.setFormatter(formatter)
    if _default_handler not in logger.handlers:
        logger.addHandler(_default_handler)

    return logger
