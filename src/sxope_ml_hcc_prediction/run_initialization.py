import argparse
import datetime

import pandas as pd
from pp_ds_ml_base.data.datasets.base import BaseDataset
from pp_ds_ml_base.etl.ml_inference_metadata import InferenceGeneralBQScheme
from pp_ds_ml_base.features.features import Features

from sxope_ml_hcc_prediction import __version__
from sxope_ml_hcc_prediction.app_config import ModelPhase, app_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_version")
    parser.add_argument("--run_id")
    parser.add_argument("--run_created_at", type=lambda ts: datetime.datetime.utcfromtimestamp(float(ts)))
    parser.add_argument("--dataset_id")
    parser.add_argument("--predicted_for_dos", type=datetime.date.fromisoformat)
    args = parser.parse_args()

    dataset = BaseDataset(
        env_name=app_config.env_name,
        project=app_config.google_project,
        model_name=app_config.model_name,
        bucket_name=app_config.gcs_bucket,
        local_data_path=app_config.get_data_path(phase=ModelPhase.inference),
        credentials_path=app_config.gcloud_secret_json,
        source_code_version=__version__,
        dataset_id_hex=args.dataset_id,
    )
    dataset.upload_to_bq()

    scheme = InferenceGeneralBQScheme(
        env_name=app_config.env_name,
        project=app_config.google_project,
        prediction_result=Features(pd.DataFrame()),
        feature_importancies=None,
        model_version=args.model_version,
        model_name=app_config.model_name,
        predicted_for_dos=args.predicted_for_dos,
        dataset_id_hex=args.dataset_id,
        run_id_hex=args.run_id,
        run_source_id=1,
        run_created_at=args.run_created_at,
        source_code_version=__version__,
    )
    scheme.upload_run_info()


if __name__ == "__main__":
    main()
