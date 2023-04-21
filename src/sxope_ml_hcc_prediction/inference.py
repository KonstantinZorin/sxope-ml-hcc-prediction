import argparse
import datetime
import glob
import logging
import os
import pathlib
import pickle
import typing
import uuid
import warnings

import attr
import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.entities.model_registry.model_version import ModelVersion
from pp_ds_ml_base.config.explainability import CaptumExplainerConfig
from pp_ds_ml_base.config.model import BaseModelConfig
from pp_ds_ml_base.explainability.captum_explain import CaptumAttributionExplainer
from pp_ds_ml_base.features.features import EntityMappedDataset, Features
from pp_ds_ml_base.models.base import BaseModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from sxope_ml_hcc_prediction.app_config import AppConfig
from sxope_ml_hcc_prediction.dataset.config import OnlineDataset, PostProcessor
from sxope_ml_hcc_prediction.models.config import MLPAttentionConfig
from sxope_ml_hcc_prediction.models.model import MLPAttentionModel
from sxope_ml_hcc_prediction.models.unified_db import (
    GeneralBQScheme,
    InferenceGeneralBQScheme,
)
from sxope_ml_hcc_prediction.utils.handy_funcs import lowest_level

logger = logging.getLogger("inference")
warnings.filterwarnings("ignore")


@attr.s(auto_attribs=True)
class InferencePipeline:
    config: MLPAttentionConfig
    member_ids_hexes: typing.Optional[typing.List[str]] = None
    model_version: typing.Optional[str] = None
    model: BaseModel = attr.ib(init=False)
    dataloader: DataLoader = attr.ib(init=False)
    post_processor = PostProcessor()
    selected_feature_columns: list = attr.ib(init=False)
    selected_label_columns: list = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        cols_meta_path = (
            f"{AppConfig.project_root}/src/sxope_ml_hcc_prediction/static/meta/inference/{os.environ['ENVIRONMENT_NAME']}"
        )
        self.mlflow_model = self._get_mlflow_model()
        run_id = self.mlflow_model.run_id
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/{os.environ['MLFLOW_MODEL_NAME']}/feature_columns.pkl", dst_path=cols_meta_path
        )
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/{os.environ['MLFLOW_MODEL_NAME']}/label_columns.pkl", dst_path=cols_meta_path
        )
        with open(f"{cols_meta_path}/feature_columns.pkl", "rb") as f:
            self.selected_feature_columns = pickle.load(f)
        with open(f"{cols_meta_path}/label_columns.pkl", "rb") as f:
            self.selected_label_columns = pickle.load(f)
        self.model = MLPAttentionModel(
            config=self.config,
            device="cuda" if torch.cuda.is_available() else "cpu",
            selected_feature_columns=self.selected_feature_columns,
            selected_label_columns=self.selected_label_columns,
        )
        self.model.torch_model = mlflow.pytorch.load_model(f"models:/{os.environ['MLFLOW_MODEL_NAME']}/Production")
        logger.debug("Started ETL pipeline")
        self.dataset_id = uuid.uuid4().bytes
        self.dataloader = DataLoader(
            OnlineDataset(
                selected_feature_columns=self.selected_feature_columns,
                member_ids_hexes=self.member_ids_hexes,
                dataset_id=self.dataset_id,
            ),
            batch_size=128,
            shuffle=False,
            num_workers=2,
            prefetch_factor=4,
        )
        self.dataloader.dataset.save_dataset()
        self.dataloader.dataset.upload_to_gcs()
        self.dataloader.dataset.upload_to_bq()
        logger.debug("Ended ETL pipeline")

    def _get_mlflow_model(self) -> ModelVersion:
        registered_model = mlflow.search_registered_models(filter_string=f"name = '{os.environ['MLFLOW_MODEL_NAME']}'")[0]
        found_model = None
        if self.model_version:
            for model in registered_model.latest_versions:
                if model.tags:
                    if model.tags[f"{os.environ['MLFLOW_MODEL_NAME']}.model.version"] == self.model_version:
                        found_model = model
        else:
            for model in registered_model.latest_versions:
                if model.current_stage == "Production":
                    found_model = model
        assert found_model, "Version not found"
        return found_model

    def inference_online(self) -> Features:
        y_hat = []
        for X in self.dataloader:
            y_hat.append(self.model.predict_proba(X))
        y_hat = np.concatenate(y_hat, axis=0)
        logger.info("Inference processed")
        y_hat = Features(
            pd.DataFrame(y_hat, columns=self.selected_label_columns, index=self.dataloader.dataset.data.data.index)
        )
        return y_hat

    def explain_inference(self) -> EntityMappedDataset:
        captum_config = CaptumExplainerConfig(captum_explainer_class="captum.attr.IntegratedGradients")
        explainer = CaptumAttributionExplainer(forward_func=self.model.torch_model, config=captum_config)
        self.dataloader.dataset.data.data_explanations = explainer.interpret(
            torch.Tensor(self.dataloader.dataset.data.data.to_numpy(dtype="float32")).to(self.model.device)
        )
        logger.info("Feature importance processed")
        return self.dataloader.dataset.data

    def save_general_db(self, feature_importancies: EntityMappedDataset, preds: Features) -> None:
        scheme = InferenceGeneralBQScheme(
            prediction_result=preds,
            feature_imporancies=feature_importancies,
            model_version=self.mlflow_model.tags[f"{os.environ['MLFLOW_MODEL_NAME']}.model.version"],
            predicted_for_dos=datetime.datetime.utcnow(),
        )
        scheme.upload_inference_data()


@attr.s(auto_attribs=True)
class MLFlowlessInferencePipeline:
    config: MLPAttentionConfig
    artifacts_path: str
    model_version: str
    run_source_id: int
    member_ids_hexes: typing.Optional[typing.List[str]] = None
    data_path: pathlib.PosixPath = (
        AppConfig.project_root  # type: ignore
        / "src/sxope_ml_hcc_prediction/static/data"
        / f"{os.environ['ENVIRONMENT_NAME']}/test/*.pkl"
    )
    dataloader: DataLoader = attr.ib(init=False)
    post_processor = PostProcessor()
    selected_feature_columns: list = attr.ib(init=False)
    selected_label_columns: list = attr.ib(init=False)
    dataset_id: bytes = attr.ib(init=False)
    builder: str = attr.ib(init=False)
    undefined_target_cols: pd.Index = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(f"{self.artifacts_path}/feature_columns.pkl", "rb") as f:
            self.selected_feature_columns = pickle.load(f)
        with open(f"{self.artifacts_path}/label_columns.pkl", "rb") as f:
            self.selected_label_columns = pickle.load(f)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MLPAttentionModel(
            config=self.config,
            device=device,
            selected_feature_columns=self.selected_feature_columns,
            selected_label_columns=self.selected_label_columns,
        )
        self.model.torch_model.load_state_dict(torch.load(f"{self.artifacts_path}/model_serialized.pt", map_location=device))
        self.confidence_buckets = self._get_confidence_buckets()
        logger.debug("Started ETL pipeline")
        self.dataset_id = uuid.uuid4().bytes
        self.dataloader = DataLoader(
            OnlineDataset(
                selected_feature_columns=self.selected_feature_columns,
                member_ids_hexes=self.member_ids_hexes,
                dataset_id=self.dataset_id,
            ),
            batch_size=16,
            shuffle=False,
            num_workers=1,
            prefetch_factor=4,
        )
        self.dataloader.dataset.save_dataset()
        self.dataloader.dataset.upload_to_gcs()
        self.dataloader.dataset.upload_to_bq()

        logger.debug("Ended ETL pipeline")

    def _get_confidence_buckets(self) -> pd.DataFrame:
        scheme = GeneralBQScheme()
        model_version_id = scheme._get_model_version_id(model_version=self.model_version)
        df = scheme._get_confidence_buckets(model_version_id=model_version_id)
        return df

    def inference_online(self) -> Features:
        y_hat = []
        for X in self.dataloader:
            y_hat.append(self.model.predict_proba(X))
        y_hat = np.concatenate(y_hat, axis=0)
        logger.info("Inference processed")
        y_hat = Features(
            pd.DataFrame(y_hat, columns=self.selected_label_columns, index=self.dataloader.dataset.data.data.index)
        )
        self.undefined_target_cols = y_hat.columns[  # type: ignore
            y_hat.columns.to_series().apply(lambda x: lowest_level(x)[-1]).isnull()  # type: ignore
        ]
        defined_target_cols = pd.Series(self.selected_label_columns)
        defined_target_cols = defined_target_cols[~(defined_target_cols.isin(self.undefined_target_cols))]
        y_hat = y_hat.select_columns(defined_target_cols)  # type: ignore
        return y_hat

    def _get_explainer_calc_masks_sliced(self, preds: Features) -> list:
        prob_threshold = self.confidence_buckets[self.confidence_buckets["confidence_level_id"] > 1][
            "bucket_start_inclusively"
        ].min()
        explainability_mask = (preds.to_data_frame() >= prob_threshold).values
        explainability_mask_sliced = []
        for i in range(int(np.ceil(explainability_mask.shape[0] / self.dataloader.batch_size))):
            explainability_mask_sliced.append(
                explainability_mask[self.dataloader.batch_size * i : self.dataloader.batch_size * (i + 1)].T.tolist()
            )
        return explainability_mask_sliced

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
        self.dataloader.dataset.data.data_explanations = X_importance
        logger.info("Feature importance processed")
        return self.dataloader.dataset.data

    def save_general_db(self, feature_importancies: EntityMappedDataset, preds: Features) -> None:
        scheme = InferenceGeneralBQScheme(
            prediction_result=preds,
            feature_imporancies=feature_importancies,
            model_version=self.model_version,
            predicted_for_dos=datetime.datetime.utcnow(),
            dataset_path=self.dataloader.dataset.gsutil_uri,
            run_source_id=self.run_source_id,
        )
        scheme.upload_inference_data()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--members_path")
    parser.add_argument("--model_version")
    parser.add_argument("--artifacts_path")
    args = parser.parse_args()
    for path in tqdm(glob.glob(args.members_path)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            members_list = pd.read_csv(path, header=None).iloc[:, 0].apply(lambda x: f"'{x}'").to_list()
            pipe = MLFlowlessInferencePipeline(
                config=BaseModelConfig.from_yaml(f"{AppConfig.project_root}/config/model.yml", "torch_model_config"),
                member_ids_hexes=members_list,
                model_version=args.model_version,
                artifacts_path=args.artifacts_path,
                run_source_id=2,
            )
            preds = pipe.inference_online()
            inference_features_and_meta = pipe.explain_inference(preds)
            pipe.save_general_db(inference_features_and_meta, preds)


if __name__ == "__main__":
    main()
