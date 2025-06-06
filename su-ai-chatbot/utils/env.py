import os
import pathlib

from dotenv import load_dotenv


def load_env():
    load_dotenv(".env.prod" if os.environ.get("prod") else ".env")
    pathlib.Path(os.environ.get("TRANSFORMERS_CACHE")).mkdir(
        parents=True, exist_ok=True
    )
    pathlib.Path(os.environ.get("HF_HUB_CACHE")).mkdir(parents=True, exist_ok=True)
