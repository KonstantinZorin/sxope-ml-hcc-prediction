import datetime
import glob
import itertools
import pathlib
import typing

import attr
import numpy as np
import pandas as pd
from pp_ds_ml_base.config.connector import BigQueryConnectorConfig
from pp_ds_ml_base.config.data_build import (
    DatasetBuildConfig,
    Environment,
    FeatureStoreBuildConfig,
)
from pp_ds_ml_base.connectors.bigquery import BigQueryConnector
from pp_ds_ml_base.data.datasets.base import BaseDataset
from pp_ds_ml_base.etl.base import BaseETLPipeline
from pp_ds_ml_base.features.features import EntityMappedDataset, Features
from torch.utils.data import Dataset

from sxope_ml_hcc_prediction.app_config import ModelPhase, app_config


@attr.s(auto_attribs=True)
class PostProcessor:
    def transform(
        self, df: typing.Union[pd.DataFrame, EntityMappedDataset, Features], selected_columns: list
    ) -> typing.Union[pd.DataFrame, EntityMappedDataset]:
        self.selected_columns = selected_columns
        if (type(df) is pd.DataFrame) or (type(df) is Features):
            df = df.align(pd.DataFrame(columns=selected_columns), join="right", axis=1, fill_value=0)[0]
            df.fillna(0, inplace=True)
        elif type(df) is EntityMappedDataset:
            for data_structure in df.meta_data_structures_enum + ["data"]:
                align_params = {"other": pd.DataFrame(columns=selected_columns), "join": "right", "axis": 1}
                if data_structure == "data":
                    align_params.update({"fill_value": 0})
                aligned_features_df = getattr(df, data_structure).values.align(**align_params)[0]
                setattr(df, data_structure, Features(aligned_features_df))
            df.data.fillna(0, inplace=True)
        return df


@attr.s(auto_attribs=True)
class RawDataExtractor:
    date_start: typing.Union[str, datetime.datetime]
    date_end: typing.Union[str, datetime.datetime]
    member_id_filter: typing.Optional[typing.List[str]] = None
    days_label_after_features: int = 365
    train_mode: bool = True
    member_id_filter_dict: dict = attr.ib(init=False)
    gbq_connector: BigQueryConnector = attr.ib(init=False)
    disease_filter: dict = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.gbq_connector = BigQueryConnector(
            BigQueryConnectorConfig(credentials_path=app_config.gcloud_secret_json, project=app_config.google_project)
        )
        if self.member_id_filter:
            self.member_id_filter_dict = {"member_id_transaction_entity_type_id": self.member_id_filter}
        else:
            self.member_id_filter_dict = {}
        self.disease_filter = {
            "sxope_is_accepted_and_not_deleted_transaction_entity_value": ["true"],
            "import_type_transaction_entity_value": [
                '"raps"',
                '"mao004"',
                '"claims"',
                '"sxope_tag_extract"',
                '"pcp_encounters"',
                '"q360_codes_captured"',
                '"q360_progress_notes"',
            ],
        }
        self.disease_filter.update(self.member_id_filter_dict)  # type: ignore

    def _render_features_config(self) -> DatasetBuildConfig:
        dataset_build_config = DatasetBuildConfig(
            feature_store_build_configs=[
                FeatureStoreBuildConfig(
                    credentials_path=app_config.gcloud_secret_json,
                    feature_entity_columns=["member_id_transaction_entity_type_id"],
                    feature_values_pivoting={
                        "columns": [
                            "hcc_code",
                        ],
                        "aggfunc": "last",
                        "values": "dummy",
                        "fill_value": 0,
                    },
                    feature_entity_id_filter_values=self.disease_filter,
                    date_start=self.date_start,
                    date_end=self.date_end,
                    data_source_service="bigquery",
                    env=app_config.env_name,
                    pipeline_name="disease_registry",
                ),
                FeatureStoreBuildConfig(
                    credentials_path=app_config.gcloud_secret_json,
                    feature_entity_columns=["member_id_transaction_entity_type_id"],
                    feature_values_pivoting={
                        "aggfunc": {
                            "patient_age_transaction_entity_value": "last",
                            "disenrollment_pending_transaction_entity_value": "last",
                            "medicaid_dual_status_code_transaction_entity_value": "last",
                            "ma_risk_score_transaction_entity_value": "last",
                            "esrd_transaction_entity_value": "last",
                            "hospice_transaction_entity_value": "last",
                            "sequential_eligibility_period_length_year_transaction_entity_value": "last",
                        }
                    },
                    feature_entity_id_filter_values=self.member_id_filter_dict,
                    date_start=self.date_start,
                    date_end=self.date_end,
                    data_source_service="bigquery",
                    env=app_config.env_name,
                    pipeline_name="members_months",
                ),
                FeatureStoreBuildConfig(
                    credentials_path=app_config.gcloud_secret_json,
                    feature_entity_columns=["member_id_transaction_entity_type_id"],
                    feature_values_pivoting={
                        "columns": [
                            "patient_gender_transaction_entity_value",
                            "is_dual_transaction_entity_value",
                            "line_of_business",
                            "bucket_acuity_transaction_entity_value",
                        ],
                        "aggfunc": {"dummy": "last"},
                    },
                    feature_entity_id_filter_values=self.member_id_filter_dict,
                    date_start=self.date_start,
                    date_end=self.date_end,
                    data_source_service="bigquery",
                    env=app_config.env_name,
                    pipeline_name="members_months",
                ),
                FeatureStoreBuildConfig(
                    credentials_path=app_config.gcloud_secret_json,
                    feature_entity_columns=["member_id_transaction_entity_type_id"],
                    feature_values_pivoting={
                        "columns": "ndc_product_pharm_classes",
                        "aggfunc": "last",
                        "values": "dummy",
                    },
                    feature_entity_id_filter_values=self.member_id_filter_dict,
                    date_start=self.date_start,
                    date_end=self.date_end,
                    data_source_service="bigquery",
                    env=app_config.env_name,
                    pipeline_name="members_prescriptions",
                ),
                FeatureStoreBuildConfig(
                    credentials_path=app_config.gcloud_secret_json,
                    feature_entity_columns=["member_id_transaction_entity_type_id"],
                    feature_values_pivoting={
                        "columns": ["loinc_num"],
                        "aggfunc": "last",
                        "values": "result_value_transaction_entity_value",
                    },
                    feature_entity_id_filter_values=self.member_id_filter_dict,
                    date_start=self.date_start,
                    date_end=self.date_end,
                    data_source_service="bigquery",
                    env=app_config.env_name,
                    pipeline_name="members_labs",
                ),
            ]
        )
        return dataset_build_config

    def _render_labels_config(self):
        label_build_config = DatasetBuildConfig(
            feature_store_build_configs=[
                FeatureStoreBuildConfig(
                    credentials_path=app_config.gcloud_secret_json,
                    feature_entity_columns=["member_id_transaction_entity_type_id"],
                    feature_values_pivoting={
                        "columns": [
                            "latest_hcc_code",
                        ],
                        "aggfunc": "last",
                        "values": "dummy",
                        "fill_value": 0,
                    },
                    feature_entity_id_filter_values=self.disease_filter,
                    date_start=self.date_end + datetime.timedelta(days=1),
                    date_end=self.date_end + datetime.timedelta(days=self.days_label_after_features),
                    data_source_service="bigquery",
                    env=app_config.env_name,
                    pipeline_name="disease_registry",
                )
            ]
        )
        return label_build_config

    def build_dataset(self) -> typing.Union[EntityMappedDataset, Features]:
        features_etl = BaseETLPipeline(self._render_features_config())
        data = features_etl.build_dataset(train_mode=self.train_mode)
        if self.train_mode:
            labels_etl = BaseETLPipeline(self._render_labels_config())
            labels = labels_etl.build_dataset(train_mode=True).values
            labels.columns = labels.columns.to_series().apply(lambda x: ("label", x)).to_list()
            labels = Features(labels)  # TODO: add rename method in Features class
            data = data.concatenate(labels)
        return data


class RepoDataset(Dataset):
    def __init__(self, path_to_data_folder: str, selected_feature_columns: list, selected_label_columns: list):
        self.paths = glob.glob(path_to_data_folder)
        self.df_lengths = []
        for file in self.paths:
            df = pd.read_pickle(file)
            self.df_lengths.append(len(df))
        self.df_lengths = list(itertools.accumulate(self.df_lengths, lambda x, y: x + y))
        self.current_path = None
        self.current_df = pd.DataFrame()
        self.selected_feature_columns = selected_feature_columns
        self.selected_label_columns = selected_label_columns
        self.post_processor = PostProcessor()

    def __len__(self) -> int:
        return self.df_lengths[-1]

    def _idx_to_path(self, idx):
        prev_df_length = 0
        for i, df_length in enumerate(self.df_lengths):
            if idx < df_length:
                return self.paths[i], idx - prev_df_length
            else:
                prev_df_length = df_length

    def __getitem__(self, idx: int) -> typing.Tuple[np.ndarray, np.ndarray]:
        asking_df_path, idx = self._idx_to_path(idx)
        if self.current_path != asking_df_path:
            self.current_path = asking_df_path
            self.current_df = pd.read_pickle(asking_df_path)
            self.X = self.post_processor.transform(self.current_df, self.selected_feature_columns)
            self.y = self.post_processor.transform(self.current_df, self.selected_label_columns)
            self.X = self.X.to_numpy(dtype="float32")
            self.y = self.y.to_numpy(dtype="float32")
        X = self.X[idx]
        y = self.y[idx]
        return X, y


@attr.s(auto_attribs=True, kw_only=True)
class OnlineDataset(Dataset, BaseDataset):

    selected_feature_columns: list

    project: str = app_config.google_project
    bucket_name: str = app_config.gcs_bucket
    model_name: str = app_config.model_name

    env_name: Environment = app_config.env_name

    credentials_path: typing.Optional[pathlib.Path] = app_config.gcloud_secret_json

    local_data_path: pathlib.Path = app_config.get_data_path(phase=ModelPhase.inference)

    member_ids_hexes: typing.Optional[typing.List[str]] = None

    date_start: datetime.datetime = attr.ib(init=False)
    date_end: typing.Optional[datetime.datetime] = None

    post_processor: PostProcessor = PostProcessor()

    data: typing.Union[pd.DataFrame, EntityMappedDataset] = attr.ib(init=False)
    X: np.ndarray = attr.ib(init=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()

        self.date_start = datetime.datetime(year=2017, month=1, day=1)
        if not self.date_end:
            self.date_end = datetime.datetime.utcnow()
        data_extractor = RawDataExtractor(
            date_start=self.date_start,
            date_end=self.date_end,
            train_mode=False,
            member_id_filter=self.member_ids_hexes,
        )
        data = data_extractor.build_dataset()
        self.data = self.post_processor.transform(data, self.selected_feature_columns)
        self.X = data.data.values.to_numpy(dtype="float32")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.X[idx]

    def save_online_dataset(self):
        super().save_dataset(
            self.data, f"data_{self.date_start.strftime('%Y_%m_%d')}__{self.date_end.strftime('%Y_%m_%d')}.pkl", "test"
        )
