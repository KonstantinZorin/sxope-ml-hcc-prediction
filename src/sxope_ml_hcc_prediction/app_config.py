import importlib.resources
import logging
import os
import typing
from enum import Enum
from pathlib import Path

import attr
from dotenv import load_dotenv
from pp_ds_ml_base.config.data_build import Environment
from pp_ds_ml_base.utils.converters import str2enum

logging.basicConfig(level=logging.INFO)


def get_project_root() -> Path:
    with importlib.resources.path("sxope_ml_hcc_prediction", "__init__.py") as src_path:
        path = src_path.parents[2]
    return path


env_file = os.getenv("ENV_FILE", f"{get_project_root()}/.env.staging")
load_dotenv(dotenv_path=env_file)


class DataType(Enum):
    train: str = "train"
    test: str = "test"
    dev: str = "dev"


class ModelPhase(Enum):
    train: str = "train"
    inference: str = "inference"


@attr.s(auto_attribs=True, frozen=True)
class AppConfig:
    project_root: Path = get_project_root()
    model_name: str = os.environ["MLFLOW_MODEL_NAME"]
    ml_flow_tracking_uri: str = os.environ["MLFLOW_TRACKING_URI"]
    gcs_bucket: str = os.environ["GCS_BUCKET"]
    env_name: Environment = attr.ib(default=os.environ["ENVIRONMENT_NAME"], converter=str2enum(Environment))  # type: ignore
    google_project: str = os.environ["GOOGLE_PROJECT"]

    def get_data_path(self, phase: ModelPhase = ModelPhase.train, data_type: typing.Optional[DataType] = None) -> Path:
        if data_type:
            path = (
                self.project_root
                / f"src/sxope_ml_hcc_prediction/static/data/{phase.value}/{self.env_name.value}/{data_type.value}"
            )
        else:
            path = self.project_root / f"src/sxope_ml_hcc_prediction/static/data/{phase.value}/{self.env_name.value}"
        return path

    @property
    def train_meta_path(self) -> Path:
        return self.project_root / f"src/sxope_ml_hcc_prediction/static/meta/train/{self.env_name.value}"

    @property
    def inference_meta_path(self) -> Path:
        return self.project_root / f"src/sxope_ml_hcc_prediction/static/meta/inference/{self.env_name.value}"

    @property
    def gcloud_secret_json(self) -> typing.Optional[Path]:
        path = self.project_root / "secrets/credentials.json"
        if not os.path.exists(path):
            logging.warning(f"Credentials path {path} does not exist. Using system credentials")
            return None
        return path

    @property
    def config_path(self) -> Path:
        return self.project_root / "config/model.yml"

    @property
    def model_path(self) -> Path:
        return self.project_root / f"models/{self.env_name.value}"


app_config = AppConfig()
