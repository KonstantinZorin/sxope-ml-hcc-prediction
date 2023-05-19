import logging
import typing

import attr
import numpy as np
import torch
from pp_ds_ml_base.models.base import BaseModel
from torch.utils.data import DataLoader

from sxope_ml_hcc_prediction.app_config import app_config
from sxope_ml_hcc_prediction.models.config import MLPAttentionConfig
from sxope_ml_hcc_prediction.models.models_torch import MLPAttentionTorch
from sxope_ml_hcc_prediction.utils.early_stoppers import EarlyStopper

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class MLPAttentionModel(BaseModel):
    config: MLPAttentionConfig
    selected_feature_columns: typing.List[tuple]
    selected_label_columns: typing.List[tuple]
    optimizer: torch.optim.Optimizer = attr.ib(init=False)
    torch_model: torch.nn.Module = attr.ib(init=False)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    running_loss: float = 0
    n_batches: int = 0
    early_stopper: EarlyStopper = EarlyStopper(not_successful_fit_limit=3)
    epoch_num: int = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.torch_model = MLPAttentionTorch(
            input_dim=len(self.selected_feature_columns),
            output_dim=len(self.selected_label_columns),
            dropout_p=self.config.dropout_p,
            layer_dim=self.config.layer_dim,
            model_dim=self.config.model_dim,
        ).to(self.device)
        self.epoch_num = 1 if app_config.env_name == "staging" else 1000

    def fit(self, train_dataloader: DataLoader, test_dataloader: DataLoader) -> None:
        self.optimizer = torch.optim.Adam(self.torch_model.parameters(), self.config.learning_rate)
        self.torch_model.train(True)
        for epoch in range(self.epoch_num):
            for X_train, y_train in train_dataloader:
                # x = torch.Tensor(X_train, dtype=torch.float32) # type: ignore
                # labels = torch.Tensor(y_train, dtype=torch.float32) # type: ignore
                x = X_train.to(self.device)
                labels = y_train.to(self.device)
                self.optimizer.zero_grad()
                pred = self.torch_model(x)
                loss = self.torch_model.loss(pred, labels)  # type: ignore
                loss.backward()
                self.optimizer.step()
                self.running_loss += loss.item()
                self.n_batches += 1
            y_hat, y_true = [], []
            for X_test, y_test in test_dataloader:
                y_hat.append(self.predict_proba(X_test))
                y_true.append(y_test)
            y_hat = np.concatenate(y_hat, axis=0)
            y_true = np.concatenate(y_true, axis=0)
            logger.info(
                f"Epoch: {epoch}. Current {self.early_stopper.runtime_metric_inst.sklearn_metric_func.__name__}: "
                f"{self.early_stopper.runtime_metric}, best: {self.early_stopper.best_metric}"
            )
            if self.early_stopper.fit_number_over_limit(y_true, y_hat):  # type: ignore
                logger.info(f"Early stop")
                break

    def predict(self, x: typing.Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if type(x) is not torch.Tensor:
            x = torch.Tensor(x, dtype=torch.float32)  # type: ignore
        x = x.to(self.device)  # type: ignore
        self.torch_model.train(False)
        with torch.no_grad():
            p = self.torch_model.forward(x)
        return p.detach().cpu().numpy()

    def predict_proba(self, x: typing.Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if type(x) is not torch.Tensor:
            x = torch.Tensor(x, dtype=torch.float32)  # type: ignore
        x = x.to(self.device)  # type: ignore
        self.torch_model.train(False)
        with torch.no_grad():
            p = self.torch_model.forward(x)
            p = torch.sigmoid(p)
        return p.detach().cpu().numpy()
