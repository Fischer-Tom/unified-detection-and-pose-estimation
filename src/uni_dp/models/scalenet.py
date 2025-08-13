from dataclasses import dataclass
from copy import deepcopy

import cv2
import numpy as np
import torch
import torchvision.models


import torch.nn.functional as F
from torch import tensor as Tensor
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

torch.autograd.set_detect_anomaly(True)


class Net(nn.Module):
    def __init__(self,feat_dim=24, pretrained=True, cats_num=6, use_hw=True):
        super().__init__()
        feature_extractor = mobilenet_v3_small(pretrained=pretrained,
                                               weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT)
        feature_extractor_bbox = mobilenet_v3_small(pretrained=pretrained,
                                                                       weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT)
        self.feat_encoder_full = nn.Sequential(feature_extractor.features, feature_extractor.avgpool, nn.Flatten())
        self.feat_encoder_bbox = nn.Sequential(feature_extractor_bbox.features, feature_extractor_bbox.avgpool, nn.Flatten())

        in_dim = feature_extractor_bbox.features[-1].out_channels * 2

        self.drop = nn.Dropout(p=0.2, inplace=True)

        self.line1 = nn.Linear(in_dim, 128)
        self.line2 = nn.Linear(128 + cats_num, feat_dim)
        self.relu = nn.ReLU(inplace=True)
        self.use_hw = use_hw
        self.cats_num = cats_num
        if self.use_hw:
            feat_dim += 2

        # initialize the line3 to all zeros
        self.line3 = nn.Linear(feat_dim + cats_num,1 )
        nn.init.zeros_(self.line3.weight)
        nn.init.zeros_(self.line3.bias)
        
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten())


    def forward(self,img,roi_imgs,label,roi_wh):
        one_hot = F.one_hot(label, num_classes=self.cats_num).float()

        feat_roi = self.feat_encoder_bbox(roi_imgs)
        feat_roi = self.drop(feat_roi)

        feat_full = self.feat_encoder_full(img)
        feat_full = self.drop(feat_full)

        feat = torch.cat([feat_roi,feat_full], dim=1)

        x = self.line1(feat)
        x = self.relu(x)
        x = torch.cat([x, one_hot], dim=1)
        x = self.line2(x)
        x = self.relu(x)
        x = torch.cat([x, one_hot], dim=1)
        hw = roi_wh / 100
        if self.use_hw:
            x = torch.cat([x,hw], dim=1)

        resi_scale = self.line3(x).squeeze()
        return resi_scale

class ScaleNet(nn.Module):

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.current_sample = None
        self.dataset_name = None
        self.net = Net()
        self.optimizer = None
        self.DZI_PAD_SCALE = 1.5


    def __str__(self):
        return "ScaleNet"


    def load_state(self, checkpoint: dict) -> None:
        self.net.load_state_dict(checkpoint)
        

    def _to_device(self, sample: Tensor, device="cuda"):
        img, img_label, R, T = (
            sample["img"].to(device),
            sample["label"].to(device),
            sample["R"].to(device),
            sample["T"].to(device),
        )
        K = sample["K"].to(device) if "K" in sample else None
        idx = 0

        if "obj_to_im_idx" in sample:
            obj_to_im_idx = sample["obj_to_im_idx"].to(device)
        else:
            obj_to_im_idx = None

        if "bbox" in sample:
            bbox = sample["bbox"].to(device)
        else:
            bbox = None

        if "size" in sample:
            size = sample["size"].to(device)
        else:
            size = None

        if "scale" in sample:
            scale = sample["scale"].to(device)
        else:
            scale = None

        if "mean_scale" in sample:
            mean_scale = sample["mean_scale"].to(device)
        else:
            mean_scale = None
      

        return (
            img,
            img_label,
            idx,
            R,
            T,
            obj_to_im_idx,
            K,
            bbox,
            size,
            scale,
            mean_scale
        )

   
    @torch.no_grad()
    def forward_inference(self, sample: tuple) -> dict[str, Tensor]:
        ( 
            full_img,
            labels,
            roi_imgs,
            mean_scales,
            roi_hws
        ) = sample

        resi_scale = self.net(full_img,roi_imgs,labels,roi_hws)
        scale = resi_scale + mean_scales

        return scale
    
    def set_current_pad_index(self, n_seen_classes: int) -> None:
        pass
    