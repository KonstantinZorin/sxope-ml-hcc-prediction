import datetime
import os
import typing
import uuid

import attr
from dateutil.relativedelta import relativedelta  # type: ignore
from pp_ds_ml_base.config.connector import CloudStorageConnectorConfig
from pp_ds_ml_base.connectors.cloud_storage import CloudStorageConnector
from tqdm import tqdm

from sxope_ml_hcc_prediction.app_config import AppConfig
from sxope_ml_hcc_prediction.dataset.config import RawDataExtractor
from sxope_ml_hcc_prediction.models.unified_db import DatasetGeneralBQScheme


@attr.s(auto_attribs=True)
class OfflineDatasetBuilder:
    gcs_data_path: str = attr.ib(init=False)
    local_data_path: str = attr.ib(init=False)
    gsutil_uri: str = attr.ib(init=False)
    dataset_id: typing.Optional[str] = None
    train_mode: bool = True
    gcs_connector: CloudStorageConnector = CloudStorageConnector(
        CloudStorageConnectorConfig(
            credentials_path=AppConfig.project_root / "secrets/credentials.json",  # type: ignore
            project=os.environ["GOOGLE_PROJECT"],
            bucket_name=os.environ["GCS_BUCKET"],
        )
    )

    def __attrs_post_init__(self) -> None:
        if not self.dataset_id:
            self.dataset_id = uuid.uuid4().hex
        if self.train_mode:
            local_path_direction = "train"
        else:
            local_path_direction = "inference"
        self.local_data_path = (
            f"{AppConfig.project_root}/src/sxope_ml_hcc_prediction/static/data/"
            f"{local_path_direction}/{os.environ['ENVIRONMENT_NAME']}"
        )
        self.gcs_data_path = f"datasets/{os.environ['MLFLOW_MODEL_NAME']}/{self.dataset_id}/"
        self.gsutil_uri = f"gs://{os.environ['GCS_BUCKET']}/{self.gcs_data_path}"

    def prepare_dataset(
        self,
        train_mode: bool,
        date_start: datetime.datetime,
        date_end: typing.Optional[datetime.datetime] = None,
        **extractor_kwargs: typing.Any,
    ) -> None:
        if date_end:
            date_ends = [date_end]
        elif os.environ["ENVIRONMENT_NAME"] == "production":
            months_amount = relativedelta(datetime.datetime.utcnow() - relativedelta(years=1), date_start)
            months_amount = months_amount.years * 12 + months_amount.months
            date_ends = [date_start + relativedelta(months=i) for i in range(1, months_amount + 1)]
        else:
            date_ends = [datetime.datetime(year=2023, month=2, day=1)]
        for i, date_end_ in tqdm(enumerate(date_ends), desc="Period"):
            data_extractor = RawDataExtractor(
                date_start=date_start, date_end=date_end_, train_mode=train_mode, **extractor_kwargs
            )
            data = data_extractor.build_dataset()
            if i == len(date_ends) - 1:
                dataset_type = "test"
            else:
                dataset_type = "train"
            with open(
                f"{self.local_data_path}"
                f"/{dataset_type}/data_{date_start.strftime('%Y_%m_%d')}__{date_end_.strftime('%Y_%m_%d')}.pkl",
                "wb",
            ) as f:
                data.values.to_pickle(f)

    def upload_to_gcs(self) -> None:
        self.gcs_connector.bulk_data_folder_upload(
            target_path=f"{self.local_data_path}",
            upload_path=self.gcs_data_path,
        )

    def upload_to_bq(self) -> None:
        uploader = DatasetGeneralBQScheme(
            dataset_id=bytes.fromhex(self.dataset_id), dataset_path=self.gsutil_uri  # type: ignore
        )
        uploader.upload_dataset_data()


def build_dataset():
    builder = OfflineDatasetBuilder()
    builder.prepare_dataset(train_mode=True, date_start=datetime.date(year=2017, month=1, day=1))
    builder.upload_to_gcs()
    builder.upload_to_bq()


if __name__ == "__main__":
    build_dataset()
