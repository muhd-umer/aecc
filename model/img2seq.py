"""
Image to sequence helper model for ViT.
"""
import torch
import torch.nn as nn


def extract_patches(batch, patch_size):
    """
    Extract patches from a batch of images.

    Args:
        batch: batch of images
        patch_size: size of the patch to be extracted
    """
    b, c, h, w = batch.shape
    assert (
        h % patch_size[0] == 0 and w % patch_size[1] == 0
    ), "Patch size should be a factor of image height and width"

    ph, pw = patch_size  # patch height and width
    nh, nw = h // ph, w // pw  # number of patches along height and width

    batch_patches = torch.reshape(batch, (b, c, nh, ph, nw, pw))
    batch_patches = torch.permute(batch_patches, (0, 1, 2, 4, 3, 5))

    return batch_patches


class Img2Seq(nn.Module):
    """
    Image to sequence model.

    Args:
        img_size: size of the image
        patch_size: size of the patch to be extracted
        num_channels: number of channels in the image
        proj_dim: projection dimension
    """

    def __init__(self, img_size, patch_size, num_channels, proj_dim):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size

        nh, nw = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        n_tokens = nh * nw

        token_dim = patch_size[0] * patch_size[1] * num_channels
        self.linear = nn.Linear(token_dim, proj_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, proj_dim))
        self.pos_emb = nn.Parameter(torch.randn(n_tokens, proj_dim))

    def forward(self, batch):
        batch = extract_patches(batch, self.patch_size)
        b, c, nh, nw, ph, pw = batch.shape

        # Flatten the patches
        batch = torch.permute(batch, (0, 2, 3, 4, 5, 1))
        batch = torch.reshape(batch, (b, nh * nw, ph * pw * c))

        batch = self.linear(batch)
        cls_token = self.cls_token.expand(b, -1, -1)
        emb = batch + self.pos_emb
        batch = torch.cat((cls_token, emb), dim=1)

        return batch
