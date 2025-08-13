from omegaconf import DictConfig
from os.path import join

import numpy as np
from einops import rearrange
import torch
from torch.nn.functional import grid_sample
from pytorch3d.utils import cameras_from_opencv_projection


from torch import tensor as Tensor

def save_off(off_file_name, vertices, faces):
    out_string = "OFF\n"
    out_string += "%d %d 0\n" % (vertices.shape[0], faces.shape[0])
    for v in vertices:
        out_string += f"{v[0]:.16f} {v[1]:.16f} {v[2]:.16f}\n"
    for f in faces:
        out_string += "3 %d %d %d\n" % (f[0], f[1], f[2])
    with open(off_file_name, "w") as fl:
        fl.write(out_string)

def load_off(off_file_name: str, to_torch=False) -> tuple:
    with open(off_file_name) as file_handle:

        file_list = file_handle.readlines()
        n_points = int(file_list[1].split(" ")[0])
        all_strings = "".join(file_list[2 : 2 + n_points])
        array_ = np.fromstring(all_strings, dtype=np.float32, sep="\n")

        all_strings = "".join(file_list[2 + n_points :])
        array_int = np.fromstring(all_strings, dtype=np.int32, sep="\n")

    array_ = array_.reshape((-1, 3))

    if to_torch:
        return torch.from_numpy(array_), torch.from_numpy(
            array_int.reshape((-1, 4))[:, 1::],
        )
    else:
        return array_, array_int.reshape((-1, 4))[:, 1::]

def parse_meshes(cfg: DictConfig) -> tuple:
    xverts, xfaces = [], []
    for cls in cfg.dataset.classes:
        mesh_path = join(cfg.dataset.paths.mesh_path, f"{cls}.off")
        xvert, xface = load_off(mesh_path, to_torch=True)
        xverts.append(pre_process_mesh(xvert))
        xfaces.append(xface)
    return xverts, xfaces


def pre_process_mesh(verts: Tensor) -> Tensor:
    verts = torch.cat((verts[..., 0:1], verts[..., 1:2], verts[..., 2:3]), dim=-1)
    return verts

def rescale_mesh(xverts: list, sizes: Tensor, scale_only: bool = False) -> list:
    scaled_verts = []
    sizes = sizes.cpu()
    for idx, size in enumerate(sizes):
        relative_size = size
        if not scale_only:
            mesh_size = 2 * torch.amax(torch.abs(xverts[idx]), dim=0)
            relative_size = size / mesh_size
        scaled_verts.append(xverts[idx].clone() * relative_size)
    return scaled_verts

def features_to_keypoints(feats: Tensor, keypoints: Tensor, img_size: tuple) -> Tensor:
    feats = torch.stack([feats]*len(keypoints), dim=0)
    kp = (
        2
        * keypoints[..., :2]
        / (
            torch.tensor(img_size, device=keypoints.device, dtype=torch.float32)[
               None, None, ...
            ]
            - 1
        )
        - 1
    )
    out_feats = grid_sample(
        feats, kp.unsqueeze(1)[..., [1, 0]], mode="bilinear", align_corners=True, padding_mode="zeros"
    )
    return rearrange(out_feats, "b c 1 v -> b v c")

def convert_points_py3d_to_cv2(points3d, points2d):
    if isinstance(points3d, torch.Tensor):
        imagePoints = points2d[:, [1, 0]].cpu().numpy()

        objectPoints = (
            torch.cat(
                (-points3d[..., 0:1], -points3d[..., 1:2], points3d[..., 2:3]),
                dim=-1,
            )
            .cpu()
            .numpy()

        )
    else:
        imagePoints = points2d[:, [1, 0]]
        objectPoints = np.concatenate(
            [-points3d[..., 0:1], -points3d[..., 1:2], points3d[..., 2:3]], axis=-1
        )

    return objectPoints, imagePoints

def convert_RTs_from_cv2_to_py3d(Rs, Ts, Ks, img_sizes):
    Rs = torch.tensor(np.stack(Rs, axis=0))
    Ts = torch.tensor(np.stack(Ts, axis=0))

    cam = cameras_from_opencv_projection(
        Rs,
        Ts,
        Ks.repeat(len(Rs), 1, 1).to(Rs.device),
        img_sizes.repeat(len(Rs), 1),
    )

    est_R = cam.R.clone().float()
    est_R[:, :2, :] *= -1
    est_T = cam.T.clone().float()


    return list(est_R), list(est_T)