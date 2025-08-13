import torch
from torch import nn, Tensor
import torch.nn.functional as F
from omegaconf import DictConfig
from copy import deepcopy

from typing import List
import math

class AdaptedDINOBlock(nn.Module):
    def __init__(self, original_block, bottleneck_dim=64, s=0.1, p=0.1):
        super().__init__()
        dim = original_block.norm1.weight.shape[0]
        self.dino_block = original_block
        self.D = nn.Linear(dim, bottleneck_dim)
        self.U = nn.Linear(bottleneck_dim, dim)
        self.LN = nn.LayerNorm(dim, eps=1e-6)
        self.bottleneck_dim = bottleneck_dim
        self.s = s
        self.dropout = nn.Dropout(p)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.D.weight, a=math.sqrt(5))
            nn.init.zeros_(self.U.weight)
            nn.init.zeros_(self.D.bias)
            nn.init.zeros_(self.U.bias)

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """

        def attn_residual_func(x: Tensor) -> Tensor:
            return self.dino_block.ls1(self.dino_block.attn(self.dino_block.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.dino_block.ls2(self.dino_block.mlp(self.dino_block.norm2(x)))

        x = x + attn_residual_func(x)

        x_l = ffn_residual_func(x)
        x_t = self.U(self.dropout(F.gelu(self.D(self.LN(x)))))
        x = x + x_l + self.s * x_t
        return x

class DINOAdapter(nn.Module):

    def __init__(
        self,
        cfg: DictConfig
    ):
        super().__init__()
        self.cfg = deepcopy(cfg)
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval()
        del self.model.mask_token
        self.token_size = 14
        for param in self.model.parameters():
            param.requires_grad = False
        for t_layer_i, blk in enumerate(self.model.blocks):
            self.model.blocks[t_layer_i] = AdaptedDINOBlock(
                blk, bottleneck_dim=self.cfg.bottleneck_dim, s=1.0, p=0.0
            )

    def get_divisible_size(self, w, h):
        return w + (14 - w % 14), h + (14 - h % 14)

    @property
    def size_divisibility(self):
        return 14

    def forward(self, x):
        w, h = x.shape[-2:]
        dw, dh = self.get_divisible_size(w, h)
        x_inp = F.interpolate(x, size=(dw, dh))
        pw, ph = dw // self.token_size, dh // self.token_size

        feat = self.model.forward_features(x_inp)
        patch = feat["x_norm_patchtokens"]

        patch = patch.reshape(patch.shape[0], pw, ph, patch.shape[-1]).permute(
            0, 3, 1, 2
        )

        feat_list = []
        for scale in self.cfg.fpn_downsample_rate:
            new_patch = F.interpolate(patch, size=(w // scale, h // scale))
            feat_list.append(new_patch)

        return feat_list[::-1], feat["x_norm_clstoken"]

    def train(self, mode=True):
        self.model.eval()
        for block in self.model.blocks:
            if isinstance(block, AdaptedDINOBlock):
                block.train(mode)
                block.dino_block.eval()
