import logging

logger = logging.getLogger("utils.log")
logger.setLevel(logging.INFO)

if not logger.handlers:  # Prevent adding multiple handlers
    handler = logging.StreamHandler()  # or FileHandler if you want logs saved
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.propagate = False