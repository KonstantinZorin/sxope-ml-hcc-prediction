import typing

import numpy as np
from pp_ds_ml_base.metrics.base import BaseMetric


def non_binary_score(scorer: BaseMetric, y_test: np.ndarray, y_pred: np.ndarray) -> typing.List:
    column_wise_metric = []
    for i in range(y_test.shape[1]):
        if len(np.unique(y_test[:, i])) < 2:
            raise Exception("Given class has only one distinct label")
        col_score = scorer.score(y_test[:, i], y_pred[:, i])
        column_wise_metric.append(col_score)
    return column_wise_metric
