"""Shared logging configuration."""

import logging

LOG_LEVEL = logging.DEBUG

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(levelname)-5s | %(message)s",
)
