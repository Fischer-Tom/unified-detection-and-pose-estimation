from dataclasses import dataclass
from copy import deepcopy
from pyprogressivex import find6DPoses

import numpy as np
import torch
from einops import einsum, rearrange, repeat
import matplotlib.pyplot as plt

from torch import tensor as Tensor
import torch.nn as nn
import torch.nn.functional as F
import cv2

from .criterion import Criterion
from .model import Model
from src.uni_dp.libs import PoseRefiner, convert_to, convert_tensor_to, ImageVariant
from src.uni_dp.tools import convert_points_py3d_to_cv2,convert_RTs_from_cv2_to_py3d
from src.uni_dp.dataset.transforms import get_real_hw,crop_resize_by_warp_affine
from src.uni_dp.tools.evaluator import get_3d_bbox, get_bbox,transform_coordinates_3d,calculate_2d_projections,draw_bboxes,align_rotation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.nn import Module
    from os.path import Path


@dataclass
class TrainerCfg:
    pad_index: Tensor
    n_classes: int
    eps: float = float("inf")
    n_gpus: int = 1
    mean_scale: list[float] = None
    sym_ids: list[float] = None


@dataclass
class ValidationCfg:
    pnp_error_thr: float = 4.0
    refine_error_thr: float = 4.0
    similarity_thr: float = 0.7
    progx: dict = None



class UniDP(nn.Module):
    cfg: TrainerCfg
    inf_cfg: ValidationCfg
    current_pad_index: Tensor
    n_classes: int = None
    current_task_id: int = 0

    def __init__(
        self,
        neural_mesh,
        cfg: "DictConfig",
    ) -> None:
        super().__init__()
        self.net = Model(cfg.extractor)
        self.neural_mesh = neural_mesh
        self.inf_cfg = ValidationCfg(**cfg.inference)
        self.criterion = Criterion( cfg.train)
        self.pose_refiner = PoseRefiner(cfg.extractor.downsample_rate)
        self.prior =  (torch.tensor([[87, 220, 89],[165, 80, 165],[88, 128, 156],[68, 146, 72],[346, 200, 335],[146, 83, 114]]).float()/1000).norm(dim=1)

    def __str__(self):
        return f"UniDP"

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, sample, anno_m, anno_i, pad_index):
        in_img = [s["img"] for s in sample]
        in_kp_m = [s["kp"] for s in anno_m]
        in_kp_i = [s["kp"] for s in anno_i]
        n_out = self.net(in_img, in_kp_m, in_kp_i)

        loss_dict = self.criterion(
            n_out,
            sample,
            anno_m,
            anno_i,
            self.neural_mesh.weight,
            pad_index,
        )

        return loss_dict

    def normalize_memory(self):
        self.neural_mesh.normalize_memory()

    def inference(self, sample):
        in_img = [s["img"] for s in sample]
        n_out = self.net(in_img)
        m_corr, i_corr, est_nocs_m, est_nocs_i = self._match(n_out)

        visualize_prediction = False
        if visualize_prediction:
            vis_nocs_m = est_nocs_m[0].permute(2, 0, 1)
            vis_nocs_i = est_nocs_i[0].permute(2, 0, 1)

            plt.figure(figsize=(45, 7))
            plt.subplot(1, 4, 1)
            plt.imshow(convert_to(in_img[0], ImageVariant.PIL))
            plt.subplot(1, 4, 2, title="Pred NOCS Mean Shape")
            plt.imshow(convert_tensor_to(vis_nocs_m, ImageVariant.PIL))
            plt.subplot(1, 4, 3, title="Pred NOCS Instance Shape")
            plt.imshow(convert_tensor_to(vis_nocs_i, ImageVariant.PIL))

            plt.show(block=True)


        poses = self._pose_estimation(m_corr, i_corr, sample)
        scales = self._scale_estimation(poses,sample)


        return [self._pack_pose(pose,scale) for pose,scale in zip(poses,scales)]

    def _pack_pose(self, pose,scale):
        labels = [p[0] for p in pose]
        Rs = [p[1] for p in pose]
        Ts = [p[2] for p in pose]
        size_deformation = [p[3] for p in pose]

        sizes = [d * np.array(2 * torch.amax(torch.abs(self.neural_mesh.xverts[c]), dim=0))
                 for c, d in zip(labels, size_deformation)]
        return {
            "estimations": (Rs, Ts, sizes, labels,scale),
            "metrics": None
            }

    def _match(self, n_out):
        mean_shape_feats = [f["feats_m"] for f in n_out]
        mean_shape_mask = [f["bg_m"] for f in n_out]
        m_corr, est_nocs_m = self._nn_matching(mean_shape_feats, mean_shape_mask, masked=True)

        instance_shape_feats = [f["feats_i"] for f in n_out]
        instance_shape_mask = [f["bg_i"] for f in n_out]
        i_corr, est_nocs_i = self._nn_matching(instance_shape_feats, instance_shape_mask, masked=False)

        return m_corr, i_corr, est_nocs_m, est_nocs_i

    def _nn_matching(self, feats, fg_mask, masked = False):

        feats = torch.stack(feats, dim=0) if isinstance(feats, list) else feats
        fg_mask = torch.stack(fg_mask, dim=0) if isinstance(fg_mask, list) else fg_mask

        b, _, h, w = feats.shape

        norm_feats = F.normalize(self.neural_mesh.weight, p=2, dim=2)

        flat_features = rearrange(feats, "b c h w -> b c (h w)")
        score_per_pixel = einsum(norm_feats, flat_features, "k v c, b c r -> b k v r")

        v_score, idx = torch.max(score_per_pixel, dim=2)

        max_score, cat_map = torch.max(v_score, dim=1, keepdim=True)

        confidence_map = max_score >= self.inf_cfg.similarity_thr
        fg_mask = (rearrange(fg_mask, "b h w -> b 1 (h w)") > 0.5) & confidence_map


        corr3d_idx = torch.gather(idx, 1, cat_map)
        corr3d = self.neural_mesh.idx_to_vert(corr3d_idx, cat_map)
        corr2d = torch.meshgrid(
            (
                torch.arange(0, h),
                torch.arange(0, w),
            )
        )
        corr2d = repeat(torch.stack(corr2d, dim=0), "c h w -> b (h w) c", b=b).to(corr3d.device)
        cat_map = cat_map.squeeze(1)
        fg_mask = fg_mask.squeeze(1)

        pred_nocs = corr3d.clone() + 0.5
        pred_nocs=pred_nocs.clone()
        pred_nocs[~fg_mask, :] = 0
        pred_nocs = rearrange(pred_nocs, "b (h w) c -> b h w c", h=h, w=w)
        if masked:
            out = [{"corr3d": c3d[fm], "corr2d": c2d[fm], "cate_map": cm[fm]}
                   for c3d, c2d, fm, cm in
                   zip(corr3d, corr2d, fg_mask, cat_map)]
        else:
            out = [{"corr3d": c3d, "corr2d": c2d, "cate_map": cm, "mask": fm}
                   for c3d, c2d, fm, cm in zip(corr3d, corr2d, fg_mask, cat_map)]

        return out, pred_nocs

    def _pose_estimation(self, m_corr, i_corr, sample):

        K = [s["K"] for s in sample]
        m_poses = [self._multi_model_ransac(corr_dict, k) for corr_dict, k in zip(m_corr, K)]
        i_poses = [self._pose_refinement(corr_dict, m_pose, k, s["img"].shape[1:])
                   for corr_dict, m_pose, k, s in zip(i_corr, m_poses, K, sample)]


        return i_poses



    def _multi_model_ransac(self, corr_dict, K):
        detected_classes = corr_dict["cate_map"].unique()
        K = np.ascontiguousarray(K.clone().cpu().numpy())
        K[:2] /= self.net.cfg.downsample_rate
        poses = []
        for cls in detected_classes:
            cls_mask = corr_dict["cate_map"] == cls
            if cls_mask.sum() < 32:
                continue
            # Note: Sort this by classses -> fucked up right now
            object_points = corr_dict["corr3d"][cls_mask].cpu().numpy()
            image_points = corr_dict["corr2d"][cls_mask].cpu().numpy()

            object_points, image_points = convert_points_py3d_to_cv2(object_points, image_points)
            R, T, corr2ds = self._progx(object_points, image_points, K)

            poses += [(cls, r, t, c2d) for r, t, c2d in zip(R, T, corr2ds)]

        return poses

    def _progx(self, op, ip, K):
        try:
            # TMP FIX for evaluation
            poses, labeling = find6DPoses(
                np.ascontiguousarray(ip),
                np.ascontiguousarray(op),
                np.ascontiguousarray(K),
                threshold=self.inf_cfg.pnp_error_thr,
                conf=self.inf_cfg.progx.conf,
                neighborhood_ball_radius=self.inf_cfg.progx.neighborhood_ball_radius,
                spatial_coherence_weight=self.inf_cfg.progx.spatial_coherence_weight,
                maximum_tanimoto_similarity=self.inf_cfg.progx.maximum_tanimoto_similarity,
                max_iters=400,
                minimum_point_number=32,
            )
        except Exception as e:
            print(f"RANSAC failed: {e}")
            return None
        poses = poses.reshape(-1, 3, 4)
        R = poses[..., :3]
        T = poses[..., 3]
        n_pred = R.shape[0]
        # Invert x and y coordinates to revert the cv2 convention
        corr2ds = [
            ip[(labeling == i).nonzero()[0]][:,[1,0]] for i in range(n_pred)
        ]


        return R, T, corr2ds

    def _pose_refinement(self, corr_dict, m_poses, K, im_shape):
        self.pose_refiner.setup(corr_dict, m_poses, K, im_shape)
        self.pose_refiner.run(float(self.inf_cfg.refine_error_thr))
        refined_poses = self.pose_refiner.get_params()

        return refined_poses
    

    def _scale_estimation(self, poses, samples,device="cuda"):
        all_scales = []

        if self.scale_net is None:
            for pose, sample in zip(poses, samples):
                pred_labels = [p[0].item() for p in pose]
                pred_scales = self.prior[pred_labels]
                all_scales.append(pred_scales.numpy())
            return all_scales

        for pose, sample in zip(poses, samples):
            img = samples[0]["img"]
            pred_labels = [p[0].item() for p in pose]
            pred_Rs = [p[1] for p in pose]
            pred_Ts = [p[2] for p in pose]
            size_deformation = [p[3] for p in pose]
            intrinsics = sample["K"].cpu().numpy()

            pred_sizes = [d * np.array(2 * torch.amax(torch.abs(self.neural_mesh.xverts[c]), dim=0))
                 for c, d in zip(pred_labels, size_deformation)]

            mean_scales = self.prior.cuda()[pred_labels] 

            pred_Rs, pred_Ts = convert_RTs_from_cv2_to_py3d(pred_Rs, pred_Ts,sample["K"][None,...],
                                            torch.tensor(img.shape[1:]) / self.net.cfg.downsample_rate)

            pred_Rs =  torch.stack(pred_Rs, dim=0).numpy()
            pred_Ts =  torch.stack(pred_Ts, dim=0).numpy()

            pred_sRT = np.eye(4)[None, :, :].repeat(len(pred_labels), axis=0)
            pred_sRT[:, :3, :3] = np.transpose(pred_Rs @ np.array(
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32
            ), [0, 2, 1])  # [n, 3, 3]
            pred_sRT[:, :3, 3] = pred_Ts
            pred_sRT[:, :2, 3] *= -1.0  # flip the y-axis

            ori_img = img.cpu().numpy().transpose(1, 2, 0)
            roi_imgs = []
            roi_whs = []

            # prepare the roi images
            for idx,(size,transform,cls_label) in enumerate(zip(pred_sizes,pred_sRT,pred_labels)):
                if cls_label in [0,1,3]:
                    transform = align_rotation(transform)

                bbox_3d = get_3d_bbox(size, 0)
                transformed_bbox_3d = transform_coordinates_3d(bbox_3d, transform)
                projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
                # roi in the format [y1,x1, y2, x2]
                roi = [projected_bbox[:,1].min(),projected_bbox[:,0].min(), projected_bbox[:,1].max(), projected_bbox[:,0].max()]
                # get square roi 
                rmin, rmax, cmin, cmax = get_bbox(roi)
                bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
                x1, y1, x2, y2 = bbox_xyxy
                bw, bh = get_real_hw(roi,img_height=ori_img.shape[0], img_width=ori_img.shape[1])

                # here resize and crop to a fixed size 256 x 256
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                bbox_center = np.array([cx, cy])  # (w/2, h/2)
                img_scale = max(y2 - y1, x2 - x1) * self.scale_net.DZI_PAD_SCALE
                img_scale = min(img_scale, max(ori_img.shape[0], ori_img.shape[1])) * 1.0
                # roi_img: 256 x 256 x3 
                roi_img = crop_resize_by_warp_affine(
                    ori_img, bbox_center, img_scale, output_size=256, interpolation=cv2.INTER_NEAREST
                )

                roi_imgs.append(roi_img)
                roi_whs.append([bw, bh])


            # [n0, 256, 256, 3]
            pred_rois = torch.from_numpy(np.stack(roi_imgs, axis=0)).to(device)
            # [n0, 256, 256, 3] -> [n0, 3, 256, 256]
            pred_rois = pred_rois.permute(0, 3, 1, 2).float()
            roi_whs = torch.tensor(np.stack(roi_whs, axis=0), device=device).float()

            # resize the full image to 256x256, repeat for each object
            resize_full_img = F.interpolate(img[None,...],size=(256, 256), mode='bilinear', align_corners=False).repeat(len(pred_labels), 1, 1, 1)
            pred_labels = torch.tensor(pred_labels, device=device,dtype=torch.long)

            pred_scales = self.scale_net.forward_inference((resize_full_img,pred_labels, pred_rois, mean_scales,roi_whs))

            all_scales.append(pred_scales.cpu().numpy())

        
        return all_scales