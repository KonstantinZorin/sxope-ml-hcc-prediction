import datetime
import os
import typing
import uuid

import attr
import numpy as np
import pandas as pd
from pp_ds_ml_base.config.connector import BigQueryConnectorConfig
from pp_ds_ml_base.connectors.bigquery import BigQueryConnector
from pp_ds_ml_base.features.features import EntityMappedDataset, Features

from sxope_ml_hcc_prediction.app_config import AppConfig
from sxope_ml_hcc_prediction.utils.handy_funcs import lowest_level


@attr.s(auto_attribs=True)
class GeneralBQScheme:
    gbq_connector: BigQueryConnector = attr.ib(init=False)
    table_prefix: str = f"pp-ds-ml-{os.environ['ENVIRONMENT_NAME']}.ml_inference_metadata"

    def __attrs_post_init__(self):
        self.gbq_connector = BigQueryConnector(
            BigQueryConnectorConfig(
                credentials_path=f"{AppConfig.project_root}/secrets/credentials.json", project=os.environ["GOOGLE_PROJECT"]
            )
        )

    def _get_id(
        self,
        entity_name: str,
        column_with_id: str,
        column_with_name: str,
        table_name: str,
        not_found_ok: bool = False,
        col_bytes_to_hex=False,
    ) -> bytes:
        if col_bytes_to_hex:
            filter_string = f"WHERE TO_HEX({column_with_name}) = {entity_name}"
        else:
            filter_string = f"WHERE {column_with_name} = {entity_name}"
        query = f"""
        SELECT {column_with_id} FROM {self.table_prefix}.{table_name}
        {filter_string}
        """
        result_df = self.gbq_connector.bulk_data_download(query=query)
        id_result = result_df.loc[0, column_with_id] if not result_df.empty else None
        if not not_found_ok:
            assert id_result, f"{entity_name} not found in {table_name}"
        return id_result

    def _get_model_version_id(self, model_version: str):
        model_version_id = self._get_id(
            entity_name=f"'{model_version}'",
            column_with_id="version_id",
            column_with_name="version",
            table_name="model_versions",
        )
        return model_version_id

    def _get_model_type_id(self, model_type_name: str) -> bytes:
        return self._get_id(
            entity_name=model_type_name,
            column_with_id="model_type_id",
            column_with_name="model_type_name",
            table_name="model_types",
        )

    def _get_confidence_buckets(self, model_version_id: bytes) -> pd.DataFrame:
        query = f"""SELECT
        confidence_bucket_id, bucket_start_inclusively, bucket_end_exclusively, confidence_level_id
        FROM {self.table_prefix}.confidence_buckets
        WHERE TO_HEX(model_version_id) = '{bytes.hex(model_version_id)}'
        """
        confidence_buckets = self.gbq_connector.bulk_data_download(query=query)
        return confidence_buckets

    def _get_model_object_type_id(self, object_type_name: str) -> bytes:
        model_object_type_id = self._get_id(
            entity_name=object_type_name,
            column_with_id="object_type_id",
            column_with_name="object_type_name",
            table_name="model_object_types",
        )
        return model_object_type_id

    def _get_dataset_id(self, dataset_path: str) -> bytes:
        dataset_id = self._get_id(
            entity_name=f"'{dataset_path}'", column_with_id="dataset_id", column_with_name="path", table_name="datasets"
        )
        return dataset_id

    def _get_model_id(self, model_name: str) -> bytes:
        model_id = self._get_id(
            entity_name=f"'{model_name}'", column_with_id="model_id", column_with_name="model_name", table_name="models"
        )
        return model_id


@attr.s(auto_attribs=True)
class DatasetGeneralBQScheme(GeneralBQScheme):
    dataset_id: bytes = attr.ib(factory=bytes)
    dataset_path: str = attr.ib(factory=str)

    def upload_dataset_data(self) -> None:
        result_df = pd.DataFrame(
            {"dataset_id": self.dataset_id, "path": self.dataset_path, "created_at": [datetime.datetime.utcnow()]}
        )
        self.gbq_connector.bulk_data_upload(dataframe=result_df, table_id=f"{self.table_prefix}.datasets")


@attr.s(auto_attribs=True)
class TrainGeneralBQScheme(GeneralBQScheme):
    model_name: str = attr.ib(factory=str)
    model_version: str = attr.ib(factory=str)
    dataset_path: str = attr.ib(factory=str)
    mlflow_tag: str = attr.ib(factory=str)
    model_path: typing.Optional[str] = None
    model_type_name: str = "'classification'"
    object_type_name: str = "'member_id'"

    def _upload_models_versions(self) -> None:
        result_df = pd.DataFrame(
            {
                "version_id": [uuid.uuid4().bytes],
                "model_id": [self._get_model_id(self.model_name)],
                "training_dataset_id": [self._get_dataset_id(self.dataset_path)],
                "version": [self.model_version],
                "model_path": [self.model_path],
                "mlflow_tag": [self.mlflow_tag],
                "created_at": [datetime.datetime.utcnow()],
            }
        )
        self.gbq_connector.bulk_data_upload(dataframe=result_df, table_id=f"{self.table_prefix}.model_versions")

    def upload_train_data(self):
        self._upload_models_versions()


@attr.s(auto_attribs=True)
class InferenceGeneralBQScheme(GeneralBQScheme):
    prediction_result: Features = attr.ib(factory=Features)
    feature_imporancies: EntityMappedDataset = attr.ib(factory=EntityMappedDataset)
    predicted_for_dos: datetime.datetime = attr.ib(factory=datetime.datetime)  # type: ignore
    model_version: str = attr.ib(factory=str)
    model_version_id: bytes = attr.ib(init=False)
    dataset_path: typing.Optional[str] = None
    dataset_id: typing.Optional[bytes] = None
    run_source_id: int = 1
    run_id: bytes = attr.ib(init=False)
    run_created_at: datetime.datetime = attr.ib(init=False)
    unpivoted_preds: pd.DataFrame = attr.ib(init=False)
    unpivoted_feature_imporancies: pd.DataFrame = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        self.model_version_id = self._get_model_version_id(self.model_version)
        if self.dataset_path:
            self.dataset_id = self._get_dataset_id(self.dataset_path)
        else:
            self.dataset_id = None
        self.run_id = uuid.uuid4().bytes
        self.run_created_at = datetime.datetime.utcnow()
        self.confidence_buckets = self._get_confidence_buckets(model_version_id=self.model_version_id)

    def _unstack_data(
        self,
        df: Features,
        keep_index: bool = True,
        provide_meta: bool = True,
        value_column_name: typing.Optional[str] = None,
    ) -> pd.DataFrame:
        unpivoted_df = df.to_data_frame()
        if keep_index:
            unpivoted_df = unpivoted_df.unstack()
            kept_index = unpivoted_df.index.copy()
            unpivoted_df = unpivoted_df.reset_index(level=0, name="value").rename(columns={"level_0": "feature_name"})
            unpivoted_df.index = kept_index
        else:
            unpivoted_df = (
                unpivoted_df.unstack().reset_index(level=0, name="value").rename(columns={"level_0": "feature_name"})
            )
        if provide_meta:
            unpivoted_df["data_meta"] = unpivoted_df["feature_name"].apply(lambda x: lowest_level(x))
            unpivoted_df["sphere_entity_id"] = unpivoted_df["data_meta"].apply(lambda x: x[0] if len(x) == 3 else None)
            unpivoted_df["sphere_entity_type_id"] = unpivoted_df["data_meta"].apply(lambda x: x[1] if len(x) == 3 else None)
            unpivoted_df.drop(columns=["data_meta"], inplace=True)
        else:
            unpivoted_df.drop(columns=["feature_name"], inplace=True)
        if value_column_name:
            unpivoted_df.rename(columns={"value": value_column_name}, inplace=True)
        return unpivoted_df

    def create_run_df(self) -> pd.DataFrame:
        inference_runs = pd.DataFrame(
            {
                "run_id": [self.run_id],
                "model_version_id": [self.model_version_id],
                "created_at": [self.run_created_at],
                "dataset_id": [self.dataset_id],
                "run_source_id": [self.run_source_id],
            }
        )
        return inference_runs

    def _get_bucket_id_by_confidence_score(self, confidence_score: list) -> list:  # TODO: add to train
        if self.confidence_buckets.empty:
            return [None] * len(confidence_score)
        else:
            intervals = pd.arrays.IntervalArray.from_arrays(
                self.confidence_buckets["bucket_start_inclusively"],
                self.confidence_buckets["bucket_end_exclusively"],
                closed="left",
            )
            confidence_bucket_ids = [
                *map(
                    lambda x: self.confidence_buckets.loc[intervals.contains(x), "confidence_bucket_id"].iloc[0],
                    confidence_score,
                )
            ]
            return confidence_bucket_ids

    def unpivot_predictions(self) -> pd.DataFrame:
        predictions = self._unstack_data(self.prediction_result, keep_index=False, value_column_name="prediction_result")
        predictions.drop(columns=["feature_name"], inplace=True)
        predictions["prediction_id"] = [uuid.uuid4().bytes for i in range(len(predictions))]
        predictions.rename(
            columns={
                "member_id_transaction_entity_type_id": "object_id",
            }
        )
        predictions["model_version_id"] = [self.model_version_id] * predictions.shape[0]
        predictions["model_version_id_hex"] = [bytes.hex(self.model_version_id)] * predictions.shape[0]
        predictions["run_id"] = [self.run_id] * predictions.shape[0]
        predictions["run_created_at"] = [self.run_created_at] * predictions.shape[0]
        predictions["created_at"] = [datetime.datetime.utcnow()] * predictions.shape[0]
        predictions["predicted_for_dos"] = [self.predicted_for_dos] * predictions.shape[0]
        predictions["confidence_score"] = predictions["prediction_result"]
        predictions["confidence_bucket_id"] = self._get_bucket_id_by_confidence_score(
            predictions["confidence_score"].tolist()
        )
        predictions["data_owner_id"] = [bytes.fromhex("37e7d8b7a33d4d51850252572717f74b")] * predictions.shape[0]
        predictions.reset_index(names="object_id", inplace=True)
        predictions["prediction_result"] = predictions["prediction_result"].apply(
            lambda x: {"classification": x, "regression": None, "entities_extraction": None}
        )
        return predictions

    def unpivot_features_metadata(self) -> pd.DataFrame:
        flat_metadata = self._unstack_data(self.feature_imporancies.data, value_column_name="feature_value")
        meta_column_names = {"data_hashes": "transaction_hash", "data_transaction_table": "data_source_table"}
        for meta_type_name in self.feature_imporancies.meta_data_structures_enum:
            meta_type_df = self._unstack_data(
                getattr(self.feature_imporancies, meta_type_name),
                provide_meta=False,
                value_column_name=meta_column_names[meta_type_name],
            )
            flat_metadata = flat_metadata.join(meta_type_df)
        return flat_metadata

    def unpivot_feature_importancies(self, predictions: pd.DataFrame, features_meta_data: pd.DataFrame) -> pd.DataFrame:
        flat_feature_importancies = pd.DataFrame()
        prediction_ids_sliced_by_target = np.array_split(
            predictions.set_index("object_id")["prediction_id"], self.prediction_result.shape[1]
        )
        for i, imporatcies_per_target in enumerate(self.feature_imporancies.data_explanations):
            feature_importancies = pd.DataFrame(
                imporatcies_per_target,
                columns=self.feature_imporancies.data.columns.copy(),
                index=self.feature_imporancies.data.index.copy(),
            )
            feature_importancies = feature_importancies.loc[:, feature_importancies.sum(axis=0) != 0]
            feature_importancies = feature_importancies.loc[feature_importancies.sum(axis=1) != 0]
            feature_importancies = self._unstack_data(
                Features(feature_importancies), provide_meta=False, value_column_name="impact"
            )
            feature_importancies["feature_importance_id"] = uuid.uuid4().bytes
            feature_importancies = feature_importancies.join(features_meta_data)
            feature_importancies = feature_importancies.join(
                prediction_ids_sliced_by_target[i], on="member_id_transaction_entity_type_id"
            )
            feature_importancies = feature_importancies[feature_importancies["impact"] != 0]
            flat_feature_importancies = pd.concat([flat_feature_importancies, feature_importancies])
        flat_feature_importancies.reset_index(drop=True, inplace=True)
        flat_feature_importancies.loc[:, "feature_name"] = flat_feature_importancies["feature_name"].astype("str")
        return flat_feature_importancies

    def get_flat_predictions_feature_importancies(self) -> typing.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        run_df = self.create_run_df()
        unpivoted_predictions = self.unpivot_predictions()
        unpivoted_meta = self.unpivot_features_metadata()
        unpivoted_feature_importancies = self.unpivot_feature_importancies(unpivoted_predictions, unpivoted_meta)
        return run_df, unpivoted_predictions, unpivoted_feature_importancies

    def upload_inference_data(self) -> None:
        run_info, predictions, feature_importancies = self.get_flat_predictions_feature_importancies()
        self.gbq_connector.bulk_data_upload(run_info, f"{self.table_prefix}.inference_runs")
        self.gbq_connector.bulk_data_upload(predictions, f"{self.table_prefix}.model_predictions")
        self.gbq_connector.bulk_data_upload(feature_importancies, f"{self.table_prefix}.feature_importancies")
