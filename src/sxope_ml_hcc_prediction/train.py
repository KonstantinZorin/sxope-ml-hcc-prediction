import logging
import pickle

import attr
import numpy as np
import torch
from pp_ds_ml_base.config.connector import (
    BigQueryConnectorConfig,
    CloudStorageConnectorConfig,
)
from pp_ds_ml_base.config.data_build import Environment
from pp_ds_ml_base.config.metric import SKLearnMetricConfig
from pp_ds_ml_base.config.model import BaseModelConfig
from pp_ds_ml_base.connectors import BigQueryConnector
from pp_ds_ml_base.connectors.cloud_storage import CloudStorageConnector
from pp_ds_ml_base.connectors.mlflow_registry import MLFlowConnector
from pp_ds_ml_base.etl.ml_inference_metadata import TrainGeneralBQScheme
from pp_ds_ml_base.metrics.base import SKLearnMetric
from pp_ds_ml_base.models.base import BaseModel
from torch.utils.data import DataLoader

from sxope_ml_hcc_prediction import __version__
from sxope_ml_hcc_prediction.app_config import DataType, ModelPhase, app_config
from sxope_ml_hcc_prediction.dataset.config import RepoDataset
from sxope_ml_hcc_prediction.dataset.dataset import OfflineDatasetBuilder
from sxope_ml_hcc_prediction.models.config import MLPAttentionConfig
from sxope_ml_hcc_prediction.utils.scores import non_binary_score

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class TrainingPipeline:
    config: MLPAttentionConfig
    model: BaseModel = attr.ib(init=False)
    selected_feature_columns: list
    selected_label_columns: list
    model_ver: str
    dataset_id: str
    gcs_connector: CloudStorageConnector = CloudStorageConnector(
        CloudStorageConnectorConfig(
            credentials_path=app_config.gcloud_secret_json,  # type: ignore
            project=app_config.google_project,
            bucket_name=app_config.gcs_bucket,
        )
    )
    gbq_connector: BigQueryConnector = BigQueryConnector(
        BigQueryConnectorConfig(credentials_path=app_config.gcloud_secret_json, project=app_config.google_project)
    )

    data_builder: OfflineDatasetBuilder = attr.ib(init=False)
    mlflow_connector: MLFlowConnector = attr.ib(default=MLFlowConnector(app_config.model_name))
    env: Environment = Environment(app_config.env_name)

    def __attrs_post_init__(self):
        self.data_builder = OfflineDatasetBuilder(dataset_id_hex=self.dataset_id, env_name=self.env)

        self.train_dataloader = DataLoader(
            RepoDataset(
                path_to_data_folder=str(
                    app_config.get_data_path(phase=ModelPhase.train, data_type=DataType.train) / "*.pkl"
                ),
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
                path_to_data_folder=str(app_config.get_data_path(phase=ModelPhase.train, data_type=DataType.test) / "*.pkl"),
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

    def download_data(self) -> None:
        self.data_builder.download_dataset(app_config.get_data_path(phase=ModelPhase.train))

    def train(self) -> None:
        self.model.fit(self.train_dataloader, self.test_dataloader)

    def _log_mlflow_metrics(self):
        scorer = SKLearnMetric.from_config(
            config=SKLearnMetricConfig.from_yaml(app_config.config_path, "mlflow_metrics_for_logging")
        )
        y_hat, y_true = [], []
        for X_test, y_test in self.test_dataloader:
            y_hat.append(self.model.predict_proba(X_test))
            y_true.append(y_test)
        y_hat = np.concatenate(y_hat, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        self.mlflow_connector.log_metric(
            f"average {scorer.config.sklearn_metric_name}", np.mean(non_binary_score(scorer, y_true, y_hat))
        )

    def log_mlflow(self) -> None:
        self.mlflow_connector.save_torch_model(self.model.torch_model)
        self.mlflow_connector.upload_artifacts(app_config.train_meta_path / "feature_columns.pkl")

        self.mlflow_connector.upload_artifacts(app_config.train_meta_path / "label_columns.pkl")

        self._log_mlflow_metrics()

    def save_general_db(self) -> None:
        scheme = TrainGeneralBQScheme(
            env_name=app_config.env_name,
            gbq_connector=self.gbq_connector,
            model_name=app_config.model_name,
            model_version=self.model_ver,
            dataset_path=self.data_builder.gsutil_uri,
            mlflow_tags=self.mlflow_connector.get_custom_mlflow_tags(),
            source_code_version=__version__,
        )
        scheme.upload_train_data()

    def save_model(self) -> None:
        torch.save(
            self.model.torch_model.state_dict(),
            app_config.model_path / "model_serialized.pt",
        )


def main(model_ver: str, dataset_id: str):
    with open(app_config.train_meta_path / "feature_columns.pkl", "rb") as f:
        selected_feature_columns = pickle.load(f)
    with open(app_config.train_meta_path / "label_columns.pkl", "rb") as f:
        selected_label_columns = pickle.load(f)

    mlflow_connector = MLFlowConnector(app_config.model_name)
    with mlflow_connector.start_run(model_ver, dataset_id):
        pipe = TrainingPipeline(
            config=BaseModelConfig.from_yaml(app_config.config_path, "torch_model_config"),
            selected_feature_columns=selected_feature_columns,
            selected_label_columns=selected_label_columns,
            model_ver=model_ver,
            dataset_id=dataset_id,
            mlflow_connector=mlflow_connector,
        )

        pipe.download_data()
        pipe.train()
        pipe.log_mlflow()
        pipe.save_general_db()
        pipe.save_model()


if __name__ == "__main__":
    main(model_ver="v0.0.3.dev", dataset_id="70ccf4ab43c84769bda6e120838a0146")
