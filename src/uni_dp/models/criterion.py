from copy import deepcopy
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, einsum



@dataclass
class ParamCfg:
    kappa_main: float
    bg_loss_weight: float
    bg_loss: str = "L2"


class Criterion(nn.Module):
    """
    Loss criterion for pose estimation model combining contrastive loss for keypoints 
    and background segmentation loss.
    """

    def __init__(self, param_cfg):
        super().__init__()
        self.cfg = ParamCfg(**param_cfg)
        self.kappa = torch.tensor(1 / self.cfg.kappa_main)

    def forward(self, n_out, sample, anno_m, anno_i, neural_mesh_memory, pad_index):
        """
        Compute combined loss for mean shape and instance shape predictions.
        
        Args:
            n_out: Network output containing keypoint features
            sample: Input samples with ground truth labels
            anno_m: Mean shape annotations with keypoint correspondences
            anno_i: Instance shape annotations with keypoint correspondences
            neural_mesh_memory: Neural mesh memory containing vertex features
            pad_index: Boolean tensor indicating padded vertices
            
        Returns:
            Dictionary containing loss components
        """
        idx = [torch.zeros_like(l["label"][:, None]) + torch.arange(0, neural_mesh_memory.shape[1],
                                                                    device=neural_mesh_memory.device)[None, :] for l in sample]

        kp_feats_m = torch.cat([n["kp_feats_m"] for n in n_out])
        loss_m = self._loss_calc(kp_feats_m, sample, anno_m, idx, neural_mesh_memory, pad_index)
        kp_feats_i = torch.cat([n["kp_feats_i"] for n in n_out])
        loss_i = self._loss_calc(kp_feats_i, sample, anno_i, idx, neural_mesh_memory, pad_index)

        loss = 0.5 * (loss_m + loss_i)

        bg_m = [n["bg_m"] for n in n_out]
        bg_loss_m = self._bg_loss(bg_m, anno_m)
        bg_i = [n["bg_i"] for n in n_out]
        bg_loss_i = self._bg_loss(bg_i, anno_i)
        bg_loss = 0.5 * (bg_loss_m + bg_loss_i)

        loss_dict = {
            "loss": loss,
            "mask_loss": bg_loss * self.cfg.bg_loss_weight,
        }


        return loss_dict
    def _loss_calc(self, kp_feats, sample, anno, idx, neural_mesh_memory, pad_index):
        """
        Calculate contrastive loss for keypoint features against neural mesh memory.
        
        Uses cross-entropy loss with intra-class positive samples and inter-class negatives.
        Only visible keypoints contribute to the loss calculation.
        
        Args:
            kp_feats: Extracted keypoint features (B, N_kp, D)
            sample: Input samples containing labels
            anno: Annotations containing keypoint visibility
            idx: Vertex indices for correspondence
            neural_mesh_memory: Memory bank of mesh vertex features
            pad_index: Mask for padded vertices
            
        Returns:
            Scalar loss tensor
        """
        label = torch.cat([l["label"] for l in sample], dim=0)
        kp_vis = torch.cat([l["kp_vis"] for l in anno], dim=0)
        idx = torch.cat(idx, dim=0)

        # Normalize mesh features to ensure that the loss is contrastive
        # Note: keypoint features are already normalized in the model
        neural_mesh_memory = F.normalize(neural_mesh_memory, p=2, dim=-1)

        intra_sim = einsum(kp_feats, neural_mesh_memory[label], "b i v, b j v -> b i j")
        intra_pad = pad_index[label]

        inter_mesh_indices = ~F.one_hot(label, neural_mesh_memory.shape[0]).bool()
        inter_feats = rearrange(torch.stack([neural_mesh_memory[l] for l in inter_mesh_indices]), "b c v d -> b (c v) d")
        inter_sim = einsum(kp_feats, inter_feats, "b i v, b j v -> b i j")
        inter_pad = rearrange(torch.stack([pad_index[l] for l in inter_mesh_indices]), "b c v -> b (c v)")

        sim = torch.cat((intra_sim, inter_sim), dim=-1)
        pad = torch.cat((intra_pad, inter_pad), dim=-1)

        sim.masked_fill_(pad[:, None, :].repeat(1, idx.shape[1], 1), float("-inf"))
        m_sim = rearrange(sim, "b c v -> (b c) v")

        kp_vis = rearrange(kp_vis, "b c -> (b c)").type(torch.bool)
        idx = rearrange(idx, "b c -> (b c)")

        loss = F.cross_entropy(
            m_sim[kp_vis, :] * self.kappa,
            idx[kp_vis],
            reduction="mean",
        )
        return loss

    def _bg_loss(self, mask_p: list, mask_gt: list) -> Tensor:
        """
        Compute background segmentation loss between predicted and ground truth masks.
        
        Args:
            mask_p: List of predicted background masks
            mask_gt: List of ground truth background masks
            
        Returns:
            Scalar loss tensor based on configured loss type
            
        Raises:
            ValueError: If unknown loss type is specified
        """
        mask_p = torch.stack(mask_p)
        mask_gt = torch.cat([m["mask"] for m in mask_gt])
        if self.cfg.bg_loss == "L2":
            return F.mse_loss(mask_p, mask_gt)
        elif self.cfg.bg_loss == "L1":
            return F.l1_loss(mask_p, mask_gt)
        elif self.cfg.bg_loss == "dice":
            return dice_loss(mask_p, mask_gt)
        elif self.cfg.bg_loss == "bce":
            return F.binary_cross_entropy(mask_p, mask_gt)

        raise ValueError(f"Unknown loss type {self.cfg.bg_loss}")

    def cuda(self, device=None):
        super().cuda(device)
        self.kappa = self.kappa.cuda(device)
        return self

def dice_loss(inputs: Tensor, targets: Tensor, smooth: float = 1e-6) -> Tensor:
    """
    Compute Dice loss for segmentation tasks.
    
    Args:
        inputs: Predicted segmentation masks
        targets: Ground truth segmentation masks  
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice loss tensor
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()

