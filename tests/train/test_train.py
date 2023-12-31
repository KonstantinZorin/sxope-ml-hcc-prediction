import pickle
import unittest
import uuid

import mlflow
from pp_ds_ml_base.config.model import BaseModelConfig

from sxope_ml_hcc_prediction.app_config import app_config
from sxope_ml_hcc_prediction.train import TrainingPipeline


class DataBuildTests(unittest.TestCase):
    pipe: TrainingPipeline

    @classmethod
    def setUpClass(cls) -> None:
        model_ver = "v0.0.1.dev"
        dataset_id = "575e1195b3314d38aea20a63af7ddbdf"
        with open(
            app_config.train_meta_path / "feature_columns.pkl",
            "rb",
        ) as f:
            selected_feature_columns = pickle.load(f)
        with open(
            app_config.train_meta_path / "label_columns.pkl",
            "rb",
        ) as f:
            selected_label_columns = pickle.load(f)
        cls.pipe = TrainingPipeline(
            config=BaseModelConfig.from_yaml(app_config.config_path, "torch_model_config"),
            selected_feature_columns=selected_feature_columns,
            selected_label_columns=selected_label_columns,
            model_ver=model_ver,
            dataset_id=dataset_id,
        )
        mlflow.start_run(
            run_name=uuid.uuid4().hex,
            tags={
                f"{app_config.model_name}.model.version": model_ver,
                f"{app_config.model_name}.data.version": dataset_id,
            },
        )

    def test_data_download(self) -> None:
        self.pipe.download_data()

    def test_train(self) -> None:
        self.pipe.train()
        self.pipe.save_model()

    def test_general_db_upload(self) -> None:
        self.pipe.save_general_db()

    @classmethod
    def tearDownClass(cls) -> None:
        mlflow.end_run()


if __name__ == "__main__":
    unittest.main()
