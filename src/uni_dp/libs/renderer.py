from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, open_dict
from pytorch3d.renderer import (
    MeshRasterizer,
    PerspectiveCameras,
    RasterizationSettings,
)

from pytorch3d.renderer.mesh import utils as p3d_utils
from pytorch3d.structures import Meshes

from src.uni_dp.tools import rescale_mesh



@dataclass
class RenderCfg:
    image_shape: ListConfig[int]
    downsample_rate: int
    max_n: int
    name: str


class RenderEngine:
    cfg: RenderCfg
    rasterizer: MeshRasterizer

    def __init__(self, cfg: DictConfig):
        render_cfg = deepcopy(cfg.dataset)
        with open_dict(render_cfg):
            render_cfg.max_n = cfg.params.mesh.max_n
            render_cfg.downsample_rate = cfg.params.extractor.downsample_rate
        self.cfg = render_cfg

    def _setup_cameras(self, K, num_obj):
        map_shape = (
            self.cfg.image_shape[0] // self.cfg.downsample_rate,
            self.cfg.image_shape[1] // self.cfg.downsample_rate,
        )
        K = K / self.cfg.downsample_rate
        K = torch.stack([K] * num_obj, dim=0)
        M = torch.diagonal(K, dim1=-2, dim2=-1)[..., :2]
        PP = K[..., 0:2, 2]
        cameras = PerspectiveCameras(
            focal_length=M,
            principal_point=PP,
            image_size=(map_shape,),
            in_ndc=False,
        ).cuda()
        raster_settings = RasterizationSettings(
            image_size=map_shape,
            blur_radius=0,
            faces_per_pixel=1,
            bin_size=None,
            perspective_correct=True,
            cull_to_frustum=True,
            clip_barycentric_coords=True,
        )
        self.rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        )
        self.K = K


    def __call__(
        self,
        texture,
        R,
        T,
        labels,
        sizes,
        K,
        xverts,
        xfaces,
        device,
        dense=False,
        scale=None,
        scale_only=True,
    ):
        self._setup_cameras(K, len(labels))
        texture = [texture[label] for label in labels]
        verts = [xverts[label] for label in labels]
        xfaces = [xfaces[label] for label in labels]

        scale = scale[:, None].repeat(1, 3)
        # scale the mesh with the isotropic scale and size specific parameters
        if not scale_only:
            scale *= sizes
        verts = rescale_mesh(verts, scale, scale_only=scale_only)

        r_module = Rasterizer(
            verts,
            xfaces,
            texture,
            self.rasterizer,
            dense=dense,
        ).to(device)

        # projected_map: (num*obj*c*h*w) v_vis:(num_obj* n_verts)
        projected_map, v_vis, per_obj_mask, fragments = r_module(R, T, True)
        """if not scale_only:
            import matplotlib.pyplot as plt
            for m in per_obj_mask:
                plt.imshow(m.cpu().numpy())
                plt.show()"""

        # [nO,nV,3] kp:[x,y] in [W,H], W=100, H = 80
        kp = r_module.rasterizer.cameras.transform_points_screen(
            r_module.meshes.verts_padded()
        )


        padded_kp = F.pad(kp, (0, 0, 0, self.cfg.max_n - kp.shape[1]), value=-1)


        # [nO,h,w]
        obj_mask = torch.any(per_obj_mask, dim=0, keepdim=True).float()
        # [nO,nV]
        v_vis_len = [len(v) for v in verts]
        padded_vis = []
        for offset in v_vis_len:
            vis = F.pad(v_vis[:offset], (0, self.cfg.max_n - offset), value=0)

            v_vis = v_vis[offset:]
            padded_vis.append(vis)
        # [nO, max_n]
        padded_vis = torch.stack(padded_vis, dim=0).to(device)
        # mask out the keypoints outside the image
        padded_vis[
            (
                padded_kp[..., 0]
                > (self.cfg.image_shape[1] / self.cfg.downsample_rate) - 1
            )
            | (
                padded_kp[..., 1]
                > (self.cfg.image_shape[0] / self.cfg.downsample_rate) - 1
            )
            | (padded_kp[..., 0] < 0)
            | (padded_kp[..., 1] < 0)
        ] = 0
        # flip x and y to into (h,w) convention
        padded_kp = padded_kp[..., [1, 0, 2]]

        return  {"kp": padded_kp,
                "kp_vis": padded_vis,
                "mask": obj_mask,
                "rendering": projected_map,
                 }



class Rasterizer(nn.Module):
    def __init__(
        self,
        vertices,
        faces,
        memory_bank,
        rasterizer,
        dense,
        post_process=None,
        off_set_mesh=False,
    ):
        super().__init__()

        self.dense = dense
        # Convert memory features of vertices to faces
        self.face_memory = None
        self.face_to_vertex_coord = None
        self._update_memory(memory_bank=memory_bank, faces=faces)

        # Support multiple meshes at same time
        if type(vertices) == list:
            self.n_mesh = len(vertices)
            # Preprocess convert mesh Pytorch3D convention
            # Create Pytorch3D meshes
            self.meshes = Meshes(verts=vertices, faces=faces, textures=None)
        else:
            self.n_mesh = 1
            # Preprocess convert meshes to Pytorch3D convention

            # Create Pytorch3D meshes
            self.meshes = Meshes(verts=[vertices], faces=[faces], textures=None)

        self._get_face_to_vertex_coord()
        # Get the normal of the vertices [N_V, 3]
        self.verts_normals = self.meshes.verts_normals_packed()
        # Device is used during theta to R
        self.rasterizer = rasterizer
        self.post_process = post_process
        self.off_set_mesh = off_set_mesh

    def _update_memory(self, memory_bank, faces=None):
        if type(memory_bank) == list:
            if faces is None:
                faces = self.faces
            # Convert memory features of vertices to faces
            self.face_memory = torch.cat(
                [m[f.type(torch.long)] for m, f in zip(memory_bank, faces)],
                dim=0,
            )

        else:
            if faces is None:
                faces = self.faces
            # Convert memory features of vertices to faces
            self.face_memory = memory_bank[faces.type(torch.long)]

    def _get_face_to_vertex_coord(self):
        self.face_to_vertex_coord = self.meshes.verts_packed()[
            self.meshes.faces_packed()
        ].cuda()

    def to(self, *args, **kwargs):
        if "device" in kwargs.keys():
            device = kwargs["device"]
        else:
            device = args[0]
        super().to(device)

        self.meshes = self.meshes.to(device)
        return self

    def cuda(self, device=None):
        return self.to(torch.device("cuda"))

    def _mask_occluded_object(self, fragments, mask_duplicate):
        duplicate_map = (
            torch.sum((fragments.zbuf > -1), dim=0, keepdim=True, dtype=torch.float32)
            > 1
        ).repeat(len(fragments.zbuf), 1, 1, 1)
        if not mask_duplicate:
            tmp_buf = fragments.zbuf.clone()
            tmp_buf[tmp_buf < 0] = 1e6
            values, indices = torch.min(tmp_buf, dim=0, keepdim=True)
            duplicate_map.scatter_(0, indices, False)

        return duplicate_map

    def forward(
        self,
        R,
        T,
        mask_duplicate
    ):
        fragments = self.rasterizer(self.meshes, R=R, T=T)
        per_obj_mask = fragments.zbuf[..., 0] >= 0
        occlusion_map = self._mask_occluded_object(fragments, mask_duplicate)
        fragments.pix_to_face[occlusion_map] = -1
        fragments.bary_coords[occlusion_map] = -1
        fragments.dists[occlusion_map] = -1
        fragments.zbuf[occlusion_map] = -1

        frags_clone = fragments
        out_map = p3d_utils.interpolate_face_attributes(
            fragments.pix_to_face,
            fragments.bary_coords,
            self.face_memory,
        )

        out_map = out_map.squeeze(dim=3)
        out_map = out_map.transpose(3, 2).transpose(2, 1)

        pix_to_face = fragments.pix_to_face
        packed_faces = self.meshes.faces_packed()
        packed_verts = self.meshes.verts_packed()

        # [obj_num*max_n]
        vertex_visibility_map = torch.zeros(
            packed_verts.shape[0], device=packed_verts.device
        )
        visible_faces = pix_to_face.unique()
        # visible_faces = visible_faces[1:] if visible_faces[0] == -1 else visible_faces
        rm_invis = visible_faces != -1
        visible_faces = visible_faces[rm_invis]
        visible_verts_idx = packed_faces[visible_faces]
        unique_visible_verts_idx, v_counts = torch.unique(
            visible_verts_idx, return_counts=True
        )
        vertex_visibility_map[unique_visible_verts_idx] = 1.0


        return out_map, vertex_visibility_map, per_obj_mask, frags_clone


