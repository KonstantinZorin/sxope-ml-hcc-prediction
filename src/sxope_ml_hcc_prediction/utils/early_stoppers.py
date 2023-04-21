import pathlib

import attr
import numpy as np
from pp_ds_ml_base.config.metric import BaseMetricConfig
from pp_ds_ml_base.metrics.base import BaseMetric, SKLearnMetric

from sxope_ml_hcc_prediction.app_config import AppConfig
from sxope_ml_hcc_prediction.utils.scores import non_binary_score


@attr.s(auto_attribs=True)
class EarlyStopper:
    not_successful_fit_limit: int
    runtime_metric_inst: SKLearnMetric = attr.ib(init=False)
    runtime_metric: float = attr.ib(init=False)
    best_metric: float = attr.ib(init=False)
    not_successful_fit_number: int = attr.ib(init=False)
    config_path: pathlib.PosixPath = AppConfig.project_root / "config/model.yml"  # type: ignore

    def __attrs_post_init__(self) -> None:
        self.runtime_metric_inst = BaseMetric.from_config(
            config=BaseMetricConfig.from_yaml(self.config_path, "early_stopper_sklearn_metric")
        )
        self.best_metric = self.runtime_metric_inst.score([0, 1], [0.5, 0.5])
        self.runtime_metric = self.best_metric
        self.not_successful_fit_number = 0

    def fit_number_over_limit(self, y_test: np.ndarray, y_pred: np.ndarray) -> bool:
        self.runtime_metric = np.mean(non_binary_score(self.runtime_metric_inst, y_test, y_pred))  # type: ignore
        if self.runtime_metric >= self.best_metric:
            self.not_successful_fit_number = 0
            self.best_metric = self.runtime_metric
        else:
            self.not_successful_fit_number += 1
        return self.not_successful_fit_number > self.not_successful_fit_limit
