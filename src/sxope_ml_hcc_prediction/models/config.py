import attr
from pp_ds_ml_base.config.model import BaseModelConfig
from pp_ds_ml_base.utils.serde import serializable


@serializable
@attr.s(auto_attribs=True)
class MLPAttentionConfig(BaseModelConfig):
    dropout_p: float
    layer_dim: int
    model_dim: int
    learning_rate: float
    batch_size: int
    configurable_full_path: str = "sxope_ml_hcc_prediction.models.model.MLPAttentionModel"
