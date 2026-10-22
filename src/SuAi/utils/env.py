import logging
import os
import pathlib

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_env():
    print("Loading env")
    load_dotenv(".env.prod" if os.environ.get("prod") else ".env")

    for env_key in ("TRANSFORMERS_CACHE", "HF_HUB_CACHE"):
        cache_dir = os.environ.get(env_key)
        if not cache_dir:
            logger.warning("%s is not set, skipping cache directory creation", env_key)
            continue
        try:
            pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
            logger.debug("Created cache directory: %s", cache_dir)
        except OSError as e:
            logger.error("Failed to create %s=%s: %s", env_key, cache_dir, e)
