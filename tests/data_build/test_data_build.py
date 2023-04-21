import datetime
import unittest

from sxope_ml_hcc_prediction.dataset.dataset import OfflineDatasetBuilder


class DataBuildTests(unittest.TestCase):
    builder: OfflineDatasetBuilder

    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = OfflineDatasetBuilder()

    def test_offline_build(self) -> None:
        self.builder.prepare_dataset(
            train_mode=True,
            date_start=datetime.datetime(year=2017, month=1, day=1),
            member_id_filter=[
                '"fe22ff66cc5846b2b70fccd4c53efab9"',
                '"43957f806b80488590f2fd74e68777b0"',
                '"87a8487f05534d09a432e504e8f821f3"',
            ],
        )  # TODO: make assertions

    def test_gcs_upload(self) -> None:
        self.builder.upload_to_gcs()

    def test_bq_upload(self) -> None:
        self.builder.upload_to_bq()


if __name__ == "__main__":
    unittest.main()
