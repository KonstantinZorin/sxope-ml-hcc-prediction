import datetime
import pathlib
import typing

import attr
from dateutil.relativedelta import relativedelta  # type: ignore
from pp_ds_ml_base.config.data_build import Environment
from pp_ds_ml_base.data.datasets.base import BaseDataset
from tqdm import tqdm

from sxope_ml_hcc_prediction import __version__
from sxope_ml_hcc_prediction.app_config import DataType, ModelPhase, app_config
from sxope_ml_hcc_prediction.dataset.config import RawDataExtractor


@attr.s(auto_attribs=True)
class OfflineDatasetBuilder(BaseDataset):
    train_mode: bool = True

    project: str = app_config.google_project
    bucket_name: str = app_config.gcs_bucket
    model_name: str = app_config.model_name

    env_name: Environment = app_config.env_name

    credentials_path: typing.Optional[pathlib.Path] = app_config.gcloud_secret_json

    local_data_path: pathlib.Path = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if self.train_mode:
            self.model_phase = ModelPhase.train
        else:
            self.model_phase = ModelPhase.inference

        self.local_data_path = app_config.get_data_path(phase=self.model_phase)
        self.source_code_version = __version__

    def prepare_dataset(
        self,
        train_mode: bool,
        date_start: datetime.datetime,
        date_end: typing.Optional[datetime.datetime] = None,
        **extractor_kwargs: typing.Any,
    ) -> None:
        if date_end:
            date_ends = [date_end]
        elif app_config.env_name == "production":
            months_amount = relativedelta(datetime.datetime.utcnow() - relativedelta(years=1), date_start)
            months_amount = months_amount.years * 12 + months_amount.months
            date_ends = [date_start + relativedelta(months=i) for i in range(1, months_amount + 1)]
        else:
            date_ends = [datetime.datetime(year=2023, month=1, day=1), datetime.datetime(year=2023, month=2, day=1)]
        for i, date_end_ in tqdm(enumerate(date_ends), desc="Period"):
            data_extractor = RawDataExtractor(
                date_start=date_start, date_end=date_end_, train_mode=train_mode, **extractor_kwargs
            )
            data = data_extractor.build_dataset()
            if i == len(date_ends) - 1:
                dataset_type = DataType.test
            else:
                dataset_type = DataType.train

            self.save_dataset(
                data, f"data_{date_start.strftime('%Y_%m_%d')}__{date_end_.strftime('%Y_%m_%d')}.pkl", dataset_type.value
            )


def build_dataset():
    builder = OfflineDatasetBuilder()
    builder.prepare_dataset(train_mode=True, date_start=datetime.datetime(year=2017, month=1, day=1))
    builder.upload_to_gcs()
    builder.upload_to_bq()


if __name__ == "__main__":
    build_dataset()
