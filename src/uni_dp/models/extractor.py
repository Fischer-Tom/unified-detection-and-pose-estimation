# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Optional, Tuple, Union
from omegaconf import DictConfig
from einops import rearrange
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init

from dataclasses import dataclass
from .layers import DualTransformer, CrossAttentionLayer, ConvLayer, PositionEmbeddingSine
from .dino_adapter import DINOAdapter



@dataclass
class PixelDecoderCfg:
    transformer_dropout: float
    transformer_nheads: int
    transformer_dim_feedforward: int
    transformer_enc_layers: int
    transformer_pre_norm: bool
    conv_dim: int
    mask_dim: int
    norm: Optional[Union[str, Callable]] = None






class GeometricDecoder(nn.Module):

    def __init__(
        self,
        cfg: DictConfig

    ):
        super().__init__()
        self.cfg = cfg
        conv_dim = self.cfg.extractor_dim
        self.bg_input_proj = nn.Linear(768, conv_dim)
        weight_init.c2_xavier_fill(self.bg_input_proj)
        self.out_proj = ConvLayer(conv_dim, cfg.mesh_dim, kernel_size=1)
        self.transformer = DualTransformer(
            d_model=conv_dim,
            dropout=self.cfg.dropout,
            nhead=self.cfg.num_heads,
            dim_feedforward=self.cfg.ffn_dim,
            num_encoder_layers=self.cfg.enc_layers,
            num_decoder_layers=self.cfg.enc_layers,
            normalize_before=False,
        )
        N_steps = self.cfg.extractor_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        in_convs_m, in_convs_i, out_convs = [], [], []
        for i, dim in enumerate(self.cfg.fpn_dim[::-1]):
            conv3x3 = ConvLayer(conv_dim, conv_dim, kernel_size=3, stride=1,
                             padding=1, bias=False, norm=nn.GroupNorm(32,conv_dim), activation=F.relu)
            if i == 0:
                conv1x1_m = ConvLayer(768, conv_dim, kernel_size=1)
                conv1x1_i = ConvLayer(768, conv_dim, kernel_size=1)

            else:
                conv1x1_m = ConvLayer(768, conv_dim, kernel_size=1, stride=1,
                                    bias=False, norm=nn.GroupNorm(32,conv_dim))
                conv1x1_i = ConvLayer(768, conv_dim, kernel_size=1, stride=1,
                                     bias=False, norm=nn.GroupNorm(32, conv_dim))

            in_convs_m.append(conv1x1_m)
            in_convs_i.append(conv1x1_i)
            out_convs.append(conv3x3)
        self.in_convs_m = nn.ModuleList(in_convs_m)
        self.in_convs_i = nn.ModuleList(in_convs_i)
        self.out_convs = nn.ModuleList(out_convs)
        self.crs_attn_layers_m = nn.ModuleList(
            [
                CrossAttentionLayer(conv_dim, self.cfg.num_heads)
                for _ in range(len(self.cfg.fpn_dim))
            ]
        )
        self.crs_attn_layers_i = nn.ModuleList(
            [
                CrossAttentionLayer(conv_dim, self.cfg.num_heads)
                for _ in range(len(self.cfg.fpn_dim))
            ]
        )

    def forward(self, features, cls_token):
        bg_token = rearrange(self.bg_input_proj(cls_token), "b c -> 1 b c")
        for idx, x in enumerate(features):
            in_conv_m = self.in_convs_m[idx]
            in_conv_i = self.in_convs_i[idx]
            output_conv = self.out_convs[idx]
            if idx == 0:
                pos = self.pe_layer(x)
                f_m, bg_token_m, f_i, bg_token_i = self.transformer(
                    in_conv_m(x), in_conv_i(x), None, pos.clone(), pos, bg_token
                )
                y_m = output_conv(f_m)
                y_i = output_conv(f_i)
            else:
                cur_fpn_m = in_conv_m(x)
                y_m = cur_fpn_m + F.interpolate(y_m, size=cur_fpn_m.shape[-2:], mode="nearest")
                y_m = output_conv(y_m)
                cur_fpn_i = in_conv_i(x)
                y_i = cur_fpn_i + F.interpolate(y_i, size=cur_fpn_i.shape[-2:], mode="nearest")
                y_i = output_conv(y_i)
            need_weights = idx == (len(self.cfg.fpn_dim) - 1)
            pos = self.pe_layer(y_m).flatten(2).permute(2, 0, 1)
            bg_token_m, bg_map_m = self.crs_attn_layers_m[idx](
                rearrange(y_m, "b c h w -> (h w) b c"), bg_token_m, pos=pos, need_weights=need_weights
            )
            pos = self.pe_layer(y_i).flatten(2).permute(2, 0, 1)
            bg_token_i, bg_map_i = self.crs_attn_layers_i[idx](
                rearrange(y_i, "b c h w -> (h w) b c"), bg_token_i, pos=pos, need_weights=need_weights
            )

        bg_map_m = rearrange(bg_map_m, "b 1 (h w) -> b h w", h=y_m.shape[-2], w=y_m.shape[-1])
        bg_map_m = bg_map_m - torch.amin(bg_map_m, dim=[1, 2], keepdim=True)
        bg_map_m = bg_map_m / torch.amax(bg_map_m, dim=[1, 2], keepdim=True)

        bg_map_i = rearrange(bg_map_i, "b 1 (h w) -> b h w", h=y_i.shape[-2], w=y_i.shape[-1])
        bg_map_i = bg_map_i - torch.amin(bg_map_i, dim=[1, 2], keepdim=True)
        bg_map_i = bg_map_i / torch.amax(bg_map_i, dim=[1, 2], keepdim=True)

        y_m = self.out_proj(y_m)
        y_i = self.out_proj(y_i)
        out = [{"feats_m": f_m, "feats_i": f_i, "bg_m": m_m, "bg_i": m_i} for f_m, f_i, m_m, m_i in
               zip(y_m, y_i, bg_map_m, bg_map_i)]

        return out



class FeatureExtractor(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()

        self.dino_extractor = DINOAdapter(cfg)
        self.decoder = GeometricDecoder(cfg)

    def train(self, mode=True):
        super().train(mode)
        self.dino_extractor.train(mode)



    def forward(self, x):
        features, cls_token = self.dino_extractor(x)
        out = self.decoder(features, cls_token)
        return out