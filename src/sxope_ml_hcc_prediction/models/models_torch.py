import torch


class MLPAttentionTorch(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout_p: float, layer_dim: int, model_dim: int) -> None:
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.input_dim = input_dim

        self.linear1 = torch.nn.Linear(input_dim, layer_dim)
        self.linear1_reduce = torch.nn.Linear(layer_dim, model_dim)

        self.linear2_1 = torch.nn.Linear(model_dim, model_dim)
        self.linear2_2 = torch.nn.Linear(model_dim, model_dim)
        self.linear3_1 = torch.nn.Linear(model_dim, model_dim)
        self.linear3_2 = torch.nn.Linear(model_dim, model_dim)
        self.linear4_1 = torch.nn.Linear(model_dim, model_dim)
        self.linear4_2 = torch.nn.Linear(model_dim, model_dim)

        self.linear_out = torch.nn.Linear(model_dim, output_dim)

        self.ln1 = torch.nn.LayerNorm(layer_dim)

        self.ln2_1 = torch.nn.LayerNorm(model_dim)
        self.ln2_2 = torch.nn.LayerNorm(model_dim)
        self.ln3_1 = torch.nn.LayerNorm(model_dim)
        self.ln3_2 = torch.nn.LayerNorm(model_dim)
        self.ln4_1 = torch.nn.LayerNorm(model_dim)
        self.ln4_2 = torch.nn.LayerNorm(model_dim)

        self.multihead_attn_2 = torch.nn.MultiheadAttention(embed_dim=1, num_heads=1, batch_first=True)
        self.multihead_attn_3 = torch.nn.MultiheadAttention(embed_dim=1, num_heads=1, batch_first=True)
        self.multihead_attn_4 = torch.nn.MultiheadAttention(embed_dim=1, num_heads=1, batch_first=True)

        self.dropout1 = torch.nn.Dropout(p=dropout_p, inplace=False)
        self.dropout2 = torch.nn.Dropout(p=dropout_p, inplace=False)
        self.dropout3 = torch.nn.Dropout(p=dropout_p, inplace=False)
        self.dropout4 = torch.nn.Dropout(p=dropout_p, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoding
        x = self.linear1(x)
        x = self.ln1(x)
        x = torch.nn.ReLU()(x)
        x = self.linear1_reduce(x)  # reduce dim to 512
        x_hat = x

        #### Block 2 #####  # noqa: E266
        # Attention
        x_hat, _ = self.multihead_attn_2(
            x_hat.view(x_hat.shape[0], x_hat.shape[1], 1),
            x_hat.view(x_hat.shape[0], x_hat.shape[1], 1),
            x_hat.view(x_hat.shape[0], x_hat.shape[1], 1),
        )
        x_hat = torch.nn.Flatten()(x_hat)
        # Add & Norm
        x = x + x_hat
        x = self.ln2_1(x)
        x_hat = x
        # Feed Forward
        x_hat = self.linear2_1(x_hat)
        x_hat = torch.nn.ReLU()(x_hat)
        x_hat = self.linear2_2(x_hat)
        # Add & Norm
        x = x + x_hat
        x = self.ln2_2(x)
        x_hat = x
        #### End of Block 1 #####  # noqa: E266

        #### Block 3 #####  # noqa: E266
        # Attention
        x_hat, _ = self.multihead_attn_3(
            x_hat.view(x_hat.shape[0], x_hat.shape[1], 1),
            x_hat.view(x_hat.shape[0], x_hat.shape[1], 1),
            x_hat.view(x_hat.shape[0], x_hat.shape[1], 1),
        )
        x_hat = torch.nn.Flatten()(x_hat)
        # Add & Norm
        x = x + x_hat
        x = self.ln3_1(x)
        x_hat = x
        # Feed Forward
        x_hat = self.linear3_1(x_hat)
        x_hat = torch.nn.ReLU()(x_hat)
        x_hat = self.linear3_2(x_hat)
        # Add & Norm
        x = x + x_hat
        x = self.ln3_2(x)
        x_hat = x
        #### End of Block 3 #####  # noqa: E266

        #### Block 4 #####  # noqa: E266
        # Attention
        x_hat, _ = self.multihead_attn_3(
            x_hat.view(x_hat.shape[0], x_hat.shape[1], 1),
            x_hat.view(x_hat.shape[0], x_hat.shape[1], 1),
            x_hat.view(x_hat.shape[0], x_hat.shape[1], 1),
        )
        x_hat = torch.nn.Flatten()(x_hat)
        # Add & Norm
        x = x + x_hat
        x = self.ln4_1(x)
        x_hat = x
        # Feed Forward
        x_hat = self.linear4_1(x_hat)
        x_hat = torch.nn.ReLU()(x_hat)
        x_hat = self.linear4_2(x_hat)
        # Add & Norm
        x = x + x_hat
        x = self.ln4_2(x)
        x_hat = x
        #### End of Block 4 #####  # noqa: E266

        x = torch.nn.Flatten()(x)
        return self.linear_out(x)
