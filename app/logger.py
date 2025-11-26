import logging
import sys
from .config import settings
import os

logger = logging.getLogger("ml_api")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")

# console handler
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)

# file handler if configured
if settings.LOG_FILE:
    os.makedirs(os.path.dirname(settings.LOG_FILE), exist_ok=True)
    fh = logging.FileHandler(settings.LOG_FILE)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
