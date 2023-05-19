import argparse
import datetime
import glob
import logging
import pathlib
import pickle
import typing
import uuid
import warnings

import attr
import numpy as np
import pandas as pd
import torch
from mlflow.entities.model_registry.model_version import ModelVersion
from pp_ds_ml_base.config.connector import BigQueryConnectorConfig
from pp_ds_ml_base.config.data_build import Environment
from pp_ds_ml_base.config.explainability import CaptumExplainerConfig
from pp_ds_ml_base.connectors import BigQueryConnector
from pp_ds_ml_base.connectors.mlflow_registry import MLFlowConnector
from pp_ds_ml_base.etl.ml_inference_metadata import (
    GeneralBQScheme,
    InferenceGeneralBQScheme,
)
from pp_ds_ml_base.explainability.captum_explain import CaptumAttributionExplainer
from pp_ds_ml_base.features.features import EntityMappedDataset, Features
from torch.utils.data import DataLoader
from tqdm import tqdm

from sxope_ml_hcc_prediction import __version__
from sxope_ml_hcc_prediction.app_config import app_config
from sxope_ml_hcc_prediction.dataset.config import OnlineDataset
from sxope_ml_hcc_prediction.models.config import MLPAttentionConfig
from sxope_ml_hcc_prediction.models.model import MLPAttentionModel
from sxope_ml_hcc_prediction.utils.handy_funcs import lowest_level

logger = logging.getLogger("inference")
warnings.filterwarnings("ignore")


@attr.s(auto_attribs=True)
class BaseInferencePipeline:

    config: MLPAttentionConfig

    artifacts_path: typing.Union[str, pathlib.Path]

    batch_size: int
    shuffle: bool
    num_workers: int
    prefetch_factor: int

    run_id_hex: typing.Optional[str]
    run_created_at: typing.Optional[datetime.datetime]
    predicted_for_dos: typing.Optional[datetime.datetime]
    run_source_id: int
    dataset_id_hex: typing.Optional[str]

    model_name: str
    model_version: typing.Optional[str]

    member_ids_hexes: typing.Optional[typing.List[str]]

    is_historical_run: bool

    selected_feature_columns: list = attr.ib(init=False)
    selected_label_columns: list = attr.ib(init=False)
    confidence_buckets: pd.DataFrame = attr.ib(init=False)

    gbq_connector: BigQueryConnector = attr.ib(init=False)

    device: str = attr.ib(init=False)
    model: MLPAttentionModel = attr.ib(init=False)
    dataset_id: bytes = attr.ib(init=False)
    dataset: OnlineDataset = attr.ib(init=False)
    dataloader: DataLoader = attr.ib(init=False)

    env: Environment = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        self.init_bq()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.env = app_config.env_name
        self.init_columns_meta()
        self.init_model()
        self.init_model_weights()
        self.init_confidence_buckets()

        logger.debug("Started ETL pipeline")
        self.create_and_persist_dataset()
        self.init_data_loader()
        logger.debug("Ended ETL pipeline")

    def init_bq(self) -> None:
        self.gbq_connector = BigQueryConnector(
            BigQueryConnectorConfig(credentials_path=app_config.gcloud_secret_json, project=app_config.google_project)
        )

    def init_model(self) -> None:
        self.model = MLPAttentionModel(
            config=self.config,
            device=self.device,
            selected_feature_columns=self.selected_feature_columns,
            selected_label_columns=self.selected_label_columns,
        )

    def init_columns_meta(self) -> None:
        with open(f"{self.artifacts_path}/feature_columns.pkl", "rb") as f:
            self.selected_feature_columns = pickle.load(f)
        with open(f"{self.artifacts_path}/label_columns.pkl", "rb") as f:
            self.selected_label_columns = pickle.load(f)

    def init_model_weights(self) -> None:
        pass

    def init_confidence_buckets(self) -> None:
        scheme = GeneralBQScheme(source_code_version=__version__, gbq_connector=self.gbq_connector, env_name=self.env)
        model_version_id = scheme._get_model_version_id(model_version=self.model_version, model_name=self.model_name)
        self.confidence_buckets = scheme._get_confidence_buckets(model_version_id=model_version_id)

    def create_and_persist_dataset(self) -> None:
        if self.dataset_id_hex:
            self.dataset_id = bytes.fromhex(self.dataset_id_hex)
        else:
            self.dataset_id = uuid.uuid4().bytes
            self.dataset_id_hex = self.dataset_id.hex()

        self.dataset = OnlineDataset(
            dataset_id_hex=self.dataset_id_hex,  # type: ignore
            env_name=self.env,
            selected_feature_columns=self.selected_feature_columns,
            member_ids_hexes=self.member_ids_hexes,
            source_code_version=__version__,
            date_end=self.predicted_for_dos if self.is_historical_run else None,
        )
        self.dataset.save_online_dataset()
        # TODO: Persist in cloud storage part of the dataset according to partition number
        if not self.is_historical_run:
            self.dataset.upload_to_gcs()
            self.dataset.upload_to_bq().result()

    def init_data_loader(self) -> None:
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def inference_online(self) -> Features:
        y_hat = []
        for X in self.dataloader:
            y_hat.append(self.model.predict_proba(X))
        y_hat = np.concatenate(y_hat, axis=0)
        logger.info("Inference processed")
        y_hat = Features(pd.DataFrame(y_hat, columns=self.selected_label_columns, index=self.dataset.data.data.index))
        undefined_target_cols = y_hat.columns[  # type: ignore
            y_hat.columns.to_series().apply(lambda x: lowest_level(x)[-1]).isnull()  # type: ignore
        ]
        defined_target_cols = pd.Series(self.selected_label_columns)
        defined_target_cols = defined_target_cols[~(defined_target_cols.isin(undefined_target_cols))]
        y_hat = y_hat.select_columns(defined_target_cols)  # type: ignore
        return y_hat

    def save_general_db(self, feature_importancies: typing.Optional[EntityMappedDataset], preds: Features) -> None:
        scheme = InferenceGeneralBQScheme(
            gbq_connector=self.gbq_connector,
            env_name=self.env,
            prediction_result=preds,
            feature_importancies=feature_importancies,
            model_version=self.model_version,
            model_name=self.model_name,
            predicted_for_dos=self.predicted_for_dos or datetime.datetime.utcnow(),
            dataset_path=self.dataset.gsutil_uri,
            run_source_id=self.run_source_id,
            run_id_hex=self.run_id_hex,
            run_created_at=self.run_created_at,
            source_code_version=__version__,
        )
        scheme.upload_inference_data(should_upload_run_info=not self.is_historical_run)

    def explain_inference(self, preds: Features) -> EntityMappedDataset:
        captum_config = CaptumExplainerConfig(captum_explainer_class="captum.attr.IntegratedGradients")
        explainer = CaptumAttributionExplainer(forward_func=self.model.torch_model, config=captum_config)
        calc_masks = self._get_explainer_calc_masks_sliced(preds)
        X_importance = []
        for i, X in enumerate(self.dataloader):
            X_importance.append(
                explainer.interpret(X.to(self.model.device), target_count=len(preds.columns), calc_mask=calc_masks[i])
            )
        X_importance = np.concatenate(X_importance, axis=1)
        self.dataset.data.data_explanations = X_importance
        logger.info("Feature importance processed")
        return self.dataset.data

    def _get_explainer_calc_masks_sliced(self, preds: Features) -> list:
        prob_threshold = self.confidence_buckets[self.confidence_buckets["confidence_level_id"] > 1][
            "bucket_start_inclusively"
        ].min()
        explainability_mask = (preds.to_data_frame() >= prob_threshold).values
        explainability_mask_sliced = []
        for i in range(int(np.ceil(explainability_mask.shape[0] / self.dataloader.batch_size))):
            explainability_mask_sliced.append(
                explainability_mask[
                    self.dataloader.batch_size * i : self.dataloader.batch_size * (i + 1)  # type: ignore
                ].T.tolist()
            )
        return explainability_mask_sliced


@attr.s(auto_attribs=True)
class InferencePipeline(BaseInferencePipeline):

    artifacts_path = app_config.inference_meta_path

    batch_size: int = 128
    shuffle: bool = False
    num_workers: int = 2
    prefetch_factor: int = 4

    run_id_hex: typing.Optional[str] = None
    run_created_at: typing.Optional[datetime.datetime] = None
    predicted_for_dos: typing.Optional[datetime.datetime] = None
    run_source_id: int = 1
    dataset_id_hex: typing.Optional[str] = None

    model_name: str = app_config.model_name
    member_ids_hexes: typing.Optional[typing.List[str]] = None

    is_historical_run = False

    mlflow_connector: MLFlowConnector = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        self.mlflow_connector = MLFlowConnector(self.model_name)
        self._get_mlflow_model()
        super().__attrs_post_init__()

    def init_model_weights(self) -> None:
        self.model.torch_model = self.mlflow_connector.load_torch_model(self.model_version, self.device)

    def _get_mlflow_model(self) -> ModelVersion:
        if self.model_version:
            found_model = self.mlflow_connector.find_model_by_version(self.model_version)
        else:
            found_model = self.mlflow_connector.find_latest_model()
            if found_model:
                self.model_version = self.mlflow_connector.get_model_version(found_model)
        assert found_model, "Version not found"

        self.mlflow_connector.download_artifacts(found_model, "feature_columns.pkl", self.artifacts_path)
        self.mlflow_connector.download_artifacts(found_model, "label_columns.pkl", self.artifacts_path)
        return found_model


@attr.s(auto_attribs=True)
class MLFlowlessInferencePipeline(BaseInferencePipeline):

    batch_size: int = 16
    shuffle: bool = False
    num_workers: int = 1
    prefetch_factor: int = 4

    run_id_hex: typing.Optional[str] = None
    run_created_at: typing.Optional[datetime.datetime] = None
    predicted_for_dos: typing.Optional[datetime.datetime] = None
    run_source_id: int = 2
    dataset_id_hex: typing.Optional[str] = None

    is_historical_run = False

    model_name: str = app_config.model_name

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()

    def init_model_weights(self) -> None:
        self.model.torch_model.load_state_dict(
            torch.load(f"{self.artifacts_path}/model_serialized.pt", map_location=self.device)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--members_path")
    parser.add_argument("--model_version")
    parser.add_argument("--artifacts_path")
    parser.add_argument("--run_id", required=False)
    parser.add_argument("--dataset_id", required=False)
    parser.add_argument("--run_created_at", type=lambda ts: datetime.datetime.utcfromtimestamp(float(ts)), required=False)
    parser.add_argument("--predicted_for_dos", type=datetime.date.fromisoformat, required=False)
    parser.add_argument("--historical", action="store_true")
    args = parser.parse_args()
    for path in tqdm(glob.glob(args.members_path)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            members_list = pd.read_csv(path, header=None).iloc[:, 0].apply(lambda x: f"'{x}'").to_list()
            pipe = MLFlowlessInferencePipeline(
                config=MLPAttentionConfig.from_yaml(app_config.config_path, "torch_model_config"),
                member_ids_hexes=members_list,
                model_version=args.model_version,
                artifacts_path=args.artifacts_path,
                is_historical_run=args.historical,
                run_id_hex=args.run_id,
                run_created_at=args.run_created_at,
                predicted_for_dos=args.predicted_for_dos,
                dataset_id_hex=args.dataset_id,
            )
            preds = pipe.inference_online()
            if args.historical:
                inference_features_and_meta = None
            else:
                inference_features_and_meta = pipe.explain_inference(preds)
            pipe.save_general_db(inference_features_and_meta, preds)


if __name__ == "__main__":
    main()
