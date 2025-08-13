from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.uni_dp.tools import flatten_config
from src.uni_dp.tools import features_to_keypoints
from src.uni_dp.models.extractor import FeatureExtractor


@dataclass
class ModelCfg:
    name: str
    extractor_dim: int
    mesh_dim: int
    local_size: int
    pretrain: bool
    downsample_rate: int
    num_clutter: int
    interpolation: str


class Model(nn.Module):
    """
    Main model class that wraps feature extraction and provides normalized outputs
    for both training and inference modes.
    """
    cfg: ModelCfg
    l2_norm = lambda self, x, dim: F.normalize(x, p=2, dim=dim)

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        model_cfg = deepcopy(cfg)
        self.cfg = flatten_config(model_cfg)
        self.net = FeatureExtractor(self.cfg)

    def forward(self, img: Union[List[torch.Tensor], torch.Tensor], 
                kp_m: List[torch.Tensor] = None, 
                kp_i: List[torch.Tensor] = None) -> List[Dict[str, Any]]:
        if kp_m is None:
            return self._forward_inference(img)
        return self._train_step(img, kp_m, kp_i)

    @torch.inference_mode()
    def _forward_inference(self, img: Union[List[torch.Tensor], torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Forward pass for inference mode with L2 normalized features.
        
        Args:
            img: Input images as list or batched tensor
            
        Returns:
            List of dictionaries containing normalized mean/instance features and background masks
        """
        img = torch.stack(img, dim=0) if isinstance(img, list) else img
        n_out = self.net(img)
        f_m = [self.l2_norm(f["feats_m"], 0) for f in n_out]
        f_i = [self.l2_norm(f["feats_i"], 0) for f in n_out]

        out = [{"feats_m": m, "feats_i": i, "bg_m": n["bg_m"], "bg_i": n["bg_i"]} for m, i, n in
               zip(f_m, f_i, n_out)]
        return out

    def _train_step(self, img: Union[List[torch.Tensor], torch.Tensor], 
                   kp_m: List[torch.Tensor], 
                   kp_i: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Forward pass for training mode with keypoint feature extraction.
        
        Args:
            img: Input images as list or batched tensor
            kp_m: Mean shape keypoint positions  
            kp_i: Instance shape keypoint positions
            
        Returns:
            List of dictionaries containing keypoint features and background masks
        """
        if isinstance(img, list):
            img = torch.stack(img, dim=0)
        b, _, h, w = img.shape
        n_out = self.net(img)

        f = [n["feats_m"] for n in n_out]
        kp_to_f_m = self._prep_feats(f, kp_m)

        f = [n["feats_i"] for n in n_out]
        kp_to_f_i = self._prep_feats(f, kp_i)

        out = [{"kp_feats_m": k_m, "kp_feats_i": k_i, "bg_m": n["bg_m"], "bg_i": n["bg_i"]} for k_m, k_i, n in zip(kp_to_f_m, kp_to_f_i, n_out)]
        return out

    def _prep_feats(self, features: List[torch.Tensor], 
                   keypoint_positions: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Extract and normalize features at keypoint positions.
        
        Args:
            features: List of feature maps from network
            keypoint_positions: List of keypoint coordinates for each image
            
        Returns:
            List of extracted keypoint features
        """
        features = [self.l2_norm(f, dim=0) for f in features]
        kp_to_f = [features_to_keypoints(f, kp,
                                         (f.shape[-2], f.shape[-1])) for f, kp in zip(features, keypoint_positions)]


        return kp_to_f