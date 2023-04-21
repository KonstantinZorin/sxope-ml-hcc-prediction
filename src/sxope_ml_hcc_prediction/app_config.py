import importlib.resources
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)


class AppConfig:
    @classmethod
    @property
    def project_root(cls) -> Path:
        with importlib.resources.path("sxope_ml_hcc_prediction", "__init__.py") as src_path:
            path = src_path.parents[2]
        return path


env_file = os.getenv("ENV_FILE", f"{AppConfig.project_root}/.env.staging")
load_dotenv(dotenv_path=env_file)
