import logging
import os
import pathlib
import pickle
import uuid

import attr
import mlflow
import numpy as np
import torch
from pp_ds_ml_base.config.connector import CloudStorageConnectorConfig
from pp_ds_ml_base.config.metric import SKLearnMetricConfig
from pp_ds_ml_base.config.model import BaseModelConfig
from pp_ds_ml_base.connectors.cloud_storage import CloudStorageConnector
from pp_ds_ml_base.metrics.base import SKLearnMetric
from pp_ds_ml_base.models.base import BaseModel
from torch.utils.data import DataLoader

from sxope_ml_hcc_prediction.app_config import AppConfig
from sxope_ml_hcc_prediction.dataset.config import RepoDataset
from sxope_ml_hcc_prediction.dataset.dataset import OfflineDatasetBuilder
from sxope_ml_hcc_prediction.models.config import MLPAttentionConfig
from sxope_ml_hcc_prediction.models.unified_db import TrainGeneralBQScheme
from sxope_ml_hcc_prediction.utils.scores import non_binary_score

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class TrainingPipeline:
    config: MLPAttentionConfig
    model: BaseModel = attr.ib(init=False)
    selected_feature_columns: tuple
    selected_label_columns: tuple
    model_ver: str
    dataset_id: str
    data_path: pathlib.PosixPath = (
        AppConfig.project_root  # type: ignore
        / "src/sxope_ml_hcc_prediction/static/data"
        / f"{os.environ['ENVIRONMENT_NAME']}/train/*.pkl"
    )
    config_path: pathlib.PosixPath = AppConfig.project_root / "config/model.yml"  # type: ignore
    gcs_connector: CloudStorageConnector = CloudStorageConnector(
        CloudStorageConnectorConfig(
            credentials_path=AppConfig.project_root / "secrets/credentials.json",  # type: ignore
            project=os.environ["GOOGLE_PROJECT"],
            bucket_name=os.environ["GCS_BUCKET"],
        )
    )
    data_builder: OfflineDatasetBuilder = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.data_builder = OfflineDatasetBuilder(dataset_id=self.dataset_id)

        self.train_dataloader = DataLoader(
            RepoDataset(
                path_to_data_folder=str(self.data_path),
                selected_feature_columns=self.selected_feature_columns,
                selected_label_columns=self.selected_label_columns,
            ),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=1,
            prefetch_factor=2,
        )
        self.test_dataloader = DataLoader(
            RepoDataset(
                path_to_data_folder=str(self.data_path),
                selected_feature_columns=self.selected_feature_columns,
                selected_label_columns=self.selected_label_columns,
            ),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=1,
            prefetch_factor=2,
        )
        self.model = BaseModel.from_config(
            config=self.config,
            selected_feature_columns=self.selected_feature_columns,
            selected_label_columns=self.selected_label_columns,
        )

    def get_custom_mlflow_tags(self) -> dict:
        return {k: v for k, v in mlflow.active_run().data.tags.items() if os.environ["MLFLOW_MODEL_NAME"] in k}

    def download_data(self) -> None:
        self.gcs_connector.bulk_data_download(
            target_path=AppConfig.project_root
            / f"src/sxope_ml_hcc_prediction/static/data/{os.environ['ENVIRONMENT_NAME']}",  # type: ignore
            download_path=self.data_builder.gcs_data_path,
        )

    def train(self) -> None:
        self.model.fit(self.train_dataloader, self.test_dataloader)

    def _log_mlflow_metrics(self):
        scorer = SKLearnMetric.from_config(
            config=SKLearnMetricConfig.from_yaml(self.config_path, "mlflow_metrics_for_logging")
        )
        y_hat, y_true = [], []
        for X_test, y_test in self.test_dataloader:
            y_hat.append(self.model.predict_proba(X_test))
            y_true.append(y_test)
        y_hat = np.concatenate(y_hat, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        mlflow.log_metric(f"average {scorer.config.sklearn_metric_name}", np.mean(non_binary_score(scorer, y_true, y_hat)))

    def log_mlflow(self) -> None:
        mlflow.pytorch.log_model(
            pytorch_model=self.model.torch_model,
            artifact_path=os.environ["MLFLOW_MODEL_NAME"],
            registered_model_name=os.environ["MLFLOW_MODEL_NAME"],
        )
        mlflow.log_artifact(
            AppConfig.project_root  # type: ignore
            / "src/sxope_ml_hcc_prediction/static/meta/train"
            / f"{os.environ['ENVIRONMENT_NAME']}/feature_columns.pkl",
            artifact_path=os.environ["MLFLOW_MODEL_NAME"],
        )
        mlflow.log_artifact(
            AppConfig.project_root  # type: ignore
            / "src/sxope_ml_hcc_prediction/static/meta/train"
            / f"{os.environ['ENVIRONMENT_NAME']}/label_columns.pkl",
            artifact_path=os.environ["MLFLOW_MODEL_NAME"],
        )
        self._log_mlflow_metrics()

    def save_general_db(self) -> None:
        scheme = TrainGeneralBQScheme(
            model_name=os.environ["MLFLOW_MODEL_NAME"],
            model_version=self.model_ver,
            dataset_path=self.data_builder.gsutil_uri,
            mlflow_tag=str(self.get_custom_mlflow_tags()),
        )
        scheme.upload_train_data()

    def save_model(self) -> None:
        torch.save(
            self.model.torch_model.state_dict(),
            f"{AppConfig.project_root}/models/{os.environ['ENVIRONMENT_NAME']}/model_serialized.pt",
        )


def main(model_ver: str, dataset_id: str):
    with open(
        AppConfig.project_root  # type: ignore
        / "src/sxope_ml_hcc_prediction/static/meta/train"
        / f"{os.environ['ENVIRONMENT_NAME']}/feature_columns.pkl",
        "rb",
    ) as f:
        selected_feature_columns = pickle.load(f)
    with open(
        AppConfig.project_root  # type: ignore
        / "src/sxope_ml_hcc_prediction/static/meta/train"
        / f"{os.environ['ENVIRONMENT_NAME']}/label_columns.pkl",
        "rb",
    ) as f:
        selected_label_columns = pickle.load(f)
    pipe = TrainingPipeline(
        config=BaseModelConfig.from_yaml(f"{AppConfig.project_root}/config/model.yml", "torch_model_config"),
        selected_feature_columns=selected_feature_columns,
        selected_label_columns=selected_label_columns,
        model_ver=model_ver,
        dataset_id=dataset_id,
    )
    with mlflow.start_run(
        run_name=uuid.uuid4().hex,
        tags={
            f"{os.environ['MLFLOW_MODEL_NAME']}.model.version": model_ver,
            f"{os.environ['MLFLOW_MODEL_NAME']}.data.version": dataset_id,
        },
    ):
        pipe.download_data()
        pipe.train()
        pipe.log_mlflow()
        pipe.save_general_db()
        pipe.save_model()


if __name__ == "__main__":
    main(model_ver="v0.0.2.dev", dataset_id="70ccf4ab43c84769bda6e120838a0146")
