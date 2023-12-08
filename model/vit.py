"""
ViT encoder model for the denoising autoencoder (DAE).
"""
import torch.nn as nn
import torch.nn.functional as F

from .img2seq import Img2Seq


class MLP(nn.Module):
    def __init__(self, in_features, hidden_units, out_features):
        super().__init__()
        self.layers = self._build_layers(in_features, hidden_units, out_features)

    def _build_layers(self, in_features, hidden_units, out_features):
        dims = [in_features] + hidden_units + [out_features]
        layers = []

        for dim1, dim2 in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(dim1, dim2))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dims[-2], dims[-1]))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ViT(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        num_channels,
        proj_dim,
        num_heads,
        dim_feedforward,
        blocks,
        mlp_units,
        latent_dim,
    ):
        super().__init__()
        self.img2seq = Img2Seq(img_size, patch_size, num_channels, proj_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            proj_dim, num_heads, dim_feedforward, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, blocks)

        # Add a fully connected layer for classification
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.25)

        self.mlp = MLP(
            3200,  # adjust this to match the output of the transformer encoder
            mlp_units,
            latent_dim,
        )

    def forward(self, batch):
        batch = self.img2seq(batch)
        batch = self.transformer_encoder(batch)

        # Flatten the output of the transformer encoder
        batch = self.flatten(batch)
        batch = F.layer_norm(batch, batch.shape[1:])
        batch = self.dropout(batch)

        # Pass the flattened output to the classifier
        batch = self.mlp(batch)

        return batch
