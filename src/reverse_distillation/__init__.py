import sys

from loguru import logger

from . import pretrained as pretrained

# Remove the default logger sink and add a custom one
logger.remove()
log_format = (
    "<green>{time:MMM-DD HH:mm:ss}</green> | "
    "<level>{level:<7}</level> | "  # level padded to 7 chars
    "<cyan>{file.name:<14}</cyan>:"
    "<cyan>{line:>4}</cyan> | "  # line right-aligned in 4-char column
    "{message}"
)

logger.add(
    sys.stdout,
    format=log_format,
    level="INFO",
)
