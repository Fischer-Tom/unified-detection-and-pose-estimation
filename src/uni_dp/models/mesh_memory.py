from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import einsum, rearrange
from omegaconf import DictConfig
from typing import List


from src.uni_dp.tools import flatten_config


@dataclass
class MeshCfg:
    momentum: float
    mesh_dim: int
    max_n: int
    precision: str
    ema: bool
    momentum_warmup: int


class NeuralMesh(nn.Module):
    cfg: MeshCfg
    l2_normalize = lambda self, x, dim: F.normalize(x, p=2, dim=dim)
    xverts: List[Tensor]
    xfaces: List[Tensor]

    def __init__(
        self,
        xverts: List[Tensor],
        xfaces: List[Tensor],
        cfg: DictConfig,
    ):
        super().__init__()
        model_cfg = deepcopy(cfg)
        self.xverts = xverts
        self.xfaces = xfaces
        self.cfg = MeshCfg(**flatten_config(model_cfg))
        memory = torch.empty(len(xverts), self.cfg.max_n, self.cfg.mesh_dim)
        torch.nn.init.kaiming_uniform_(memory, a=1)
        self.weight = nn.Parameter(self.l2_normalize(memory, 2))
        self.momentum = self.cfg.momentum

    @property
    def n_classes(self):
        return self.weight.shape[0]

    @torch.inference_mode()
    def ema(self, n_out):
        if self.cfg.ema:
            self._updateMemory(n_out["kp_feats_m"], n_out["kp_vis_m"],n_out["label"])
            self._updateMemory(n_out["kp_feats_i"], n_out["kp_vis_i"], n_out["label"])

    @torch.no_grad()
    def normalize_memory(self, eps=1e-6):
        pass
        #self.weight = self.l2_normalize(self.weight, 2)

    def _updateMemory(self, vertices, visible, label):
        one_hot_label = F.one_hot(label, num_classes=self.n_classes).to(vertices.dtype)
        update = einsum(one_hot_label, vertices, "b k, b v c -> b k v c")
        vertex_vis = einsum(one_hot_label, visible, "b k, b v -> b k v")[...,None]

        update = torch.sum(update * vertex_vis, dim=0)
        vertex_vis = torch.sum(vertex_vis, dim=0)
        update = update / vertex_vis.clamp(min=1)

        normalized_update = self.l2_normalize(
            update,
            2)
        self.weight = self.l2_normalize(
            self.momentum * self.weight + (1 - self.momentum) * normalized_update, 2
        )



    def label_to_onehot(self, img_label, count_label):
        ret = torch.zeros(img_label.shape[0], self.n_classes, device=img_label.device)
        ret = ret.scatter_(1, img_label.unsqueeze(1), 1.0)
        for i in range(self.n_classes):
            count = count_label[i]
            if count == 0:
                continue
            ret[:, i] /= count
        return ret

    def idx_to_vert(self, corr3d_idx, cat_map):
        b, _, p = corr3d_idx.shape
        packed_verts = torch.stack(
            [
                torch.nn.functional.pad(
                    vs, (0, 0, 0, self.cfg.max_n - len(vs)), value=-1
                )
                for vs in self.xverts
            ],
        ).to(corr3d_idx.device)

        flat_c3d = corr3d_idx.reshape(-1)
        flat_cmap = cat_map.reshape(-1)

        corr3d = packed_verts[flat_cmap, flat_c3d]

        return rearrange(corr3d, "(b p) c -> b p c", b=b)

