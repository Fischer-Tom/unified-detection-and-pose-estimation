import argparse
import glob
from os.path import isfile, join
from tqdm import tqdm
import _pickle as cPickle

import numpy as np
import pytorch3d.io as IO
import json

from src.uni_dp.dataset import real275
from src.uni_dp.tools.mesh_utils import save_off

real275_cates = {"bottle": "02876657",
                "bowl": "02880940",
                "camera": "02942699",
                "can": "02946921",
                "laptop": "03642806",
                "mug": "03797390",
            }

def load_obj(file, to_tensor=False):
    verts, faces, _ = IO.load_obj(file, load_textures=False)

    if to_tensor:
        return verts, faces.verts_idx
    else:
        return verts.numpy(), faces.verts_idx.numpy()

def get_bbox(dataset, obj_path):
    if dataset.lower() == "real275":
        bbox = np.loadtxt(join(obj_path, "bbox.txt"))
        size = bbox[0] - bbox[1]
        if isfile(join(obj_path, "scale.txt")):
            scale = np.loadtxt(join(obj_path, "scale.txt"))
        else:
            scale = np.array([1.0, 1.0, 1.0])
    elif dataset.lower() == "h6d":
        size = np.loadtxt(obj_path+"_norm.txt")
        scale = np.linalg.norm(size)
        # Note: size is unnormalized in H6D dataset, we normalize here to make the code consistent with REAL275
        size = size/scale

    return size, scale

def meshify(x_range, y_range, z_range, number_vertices):
    w, h, d = x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]
    total_area = (w * h + h * d + w * d) * 2

    # On average, every vertice attach 6 edges. Each triangle has 3 edges
    mesh_size = total_area / (number_vertices * 2)

    edge_length = (mesh_size * 2) ** 0.5

    x_samples = x_range[0] + np.linspace(0, w, int(w / edge_length + 1))
    y_samples = y_range[0] + np.linspace(0, h, int(h / edge_length + 1))
    z_samples = z_range[0] + np.linspace(0, d, int(d / edge_length + 1))

    xn = x_samples.size
    yn = y_samples.size
    zn = z_samples.size

    out_vertices = []
    out_faces = []
    base_idx = 0

    for n in range(yn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[n], z_samples[0]))
    for m in range(yn - 1):
        for n in range(xn - 1):
            out_faces.append(
                (
                    base_idx + m * xn + n,
                    base_idx + m * xn + n + 1,
                    base_idx + (m + 1) * xn + n,
                ),
            )
            out_faces.append(
                (
                    base_idx + (m + 1) * xn + n + 1,
                    base_idx + m * xn + n + 1,
                    base_idx + (m + 1) * xn + n,
                ),
            )
    base_idx += yn * xn

    for n in range(yn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[n], z_samples[-1]))
    for m in range(yn - 1):
        for n in range(xn - 1):
            out_faces.append(
                (
                    base_idx + m * xn + n,
                    base_idx + m * xn + n + 1,
                    base_idx + (m + 1) * xn + n,
                ),
            )
            out_faces.append(
                (
                    base_idx + (m + 1) * xn + n + 1,
                    base_idx + m * xn + n + 1,
                    base_idx + (m + 1) * xn + n,
                ),
            )
    base_idx += yn * xn

    for n in range(zn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[0], z_samples[n]))
    for m in range(zn - 1):
        for n in range(xn - 1):
            out_faces.append(
                (
                    base_idx + m * xn + n,
                    base_idx + m * xn + n + 1,
                    base_idx + (m + 1) * xn + n,
                ),
            )
            out_faces.append(
                (
                    base_idx + (m + 1) * xn + n + 1,
                    base_idx + m * xn + n + 1,
                    base_idx + (m + 1) * xn + n,
                ),
            )
    base_idx += zn * xn

    for n in range(zn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[-1], z_samples[n]))
    for m in range(zn - 1):
        for n in range(xn - 1):
            out_faces.append(
                (
                    base_idx + m * xn + n,
                    base_idx + m * xn + n + 1,
                    base_idx + (m + 1) * xn + n,
                ),
            )
            out_faces.append(
                (
                    base_idx + (m + 1) * xn + n + 1,
                    base_idx + m * xn + n + 1,
                    base_idx + (m + 1) * xn + n,
                ),
            )
    base_idx += zn * xn

    for n in range(zn):
        for m in range(yn):
            out_vertices.append((x_samples[0], y_samples[m], z_samples[n]))
    for m in range(zn - 1):
        for n in range(yn - 1):
            out_faces.append(
                (
                    base_idx + m * yn + n,
                    base_idx + m * yn + n + 1,
                    base_idx + (m + 1) * yn + n,
                ),
            )
            out_faces.append(
                (
                    base_idx + (m + 1) * yn + n + 1,
                    base_idx + m * yn + n + 1,
                    base_idx + (m + 1) * yn + n,
                ),
            )
    base_idx += zn * yn

    for n in range(zn):
        for m in range(yn):
            out_vertices.append((x_samples[-1], y_samples[m], z_samples[n]))
    for m in range(zn - 1):
        for n in range(yn - 1):
            out_faces.append(
                (
                    base_idx + m * yn + n,
                    base_idx + m * yn + n + 1,
                    base_idx + (m + 1) * yn + n,
                ),
            )
            out_faces.append(
                (
                    base_idx + (m + 1) * yn + n + 1,
                    base_idx + m * yn + n + 1,
                    base_idx + (m + 1) * yn + n,
                ),
            )
    base_idx += zn * yn

    return np.array(out_vertices), np.array(out_faces)

def generate_mean_mesh(obj_paths, n_vertices, save_sizes=False):
    sizes = {}
    per_obj_size = []
    scales = []
    ds = "real275" if save_sizes else "h6d"
    for obj_path in obj_paths:
        splitpath = obj_path.split("/")
        if ds == "real275":
            path = "/".join(splitpath[:-1])
        elif ds == "h6d":
            path = "/".join(splitpath).replace("_norm.txt", "")
        idx = splitpath[-2]

        size, scale = get_bbox(ds, path)
        per_obj_size.append(size*scale)
        scales.append(scale)
        if save_sizes:
            sizes[idx] = size



    mean_max_pos = np.mean(np.array(per_obj_size) / 2, axis=0)
    mean_min_pos = -mean_max_pos
    out_pos = [(mean_min_pos[i], mean_max_pos[i]) for i in range(3)]
    xvert, xface = meshify(*out_pos, number_vertices=n_vertices)

    mean_scale = np.mean(np.array(scales), axis=0)
    # Rescale the Mesh
    size = 2 * np.amax(np.abs(xvert), axis=0)
    mean_scale_norm = np.linalg.norm(size)
    xvert = xvert / mean_scale_norm

    mean_scale = mean_scale * mean_scale_norm
    return sizes, np.linalg.norm(mean_scale), xvert, xface

def create_prior_meshes_real275(cfg, n_vertices=1000):
    classes = cfg.dataset.classes
    mesh_path = cfg.dataset.paths.cad_path
    out_path = cfg.dataset.paths.mesh_path
    mean_scales = {}
    sizes = {}
    for cate in tqdm(classes):
        cate_path = real275_cates[cate]
        mesh_dir = join(mesh_path, "train", cate_path, "*", "bbox.txt")

        obj_paths = glob.glob(mesh_dir)
        cate_sizes, mean_scale, xvert, xface = generate_mean_mesh(obj_paths, n_vertices=n_vertices, save_sizes=True)
        save_off(join(out_path, f"{cate}.off"), xvert, xface)
        mean_scales[cate] = mean_scale / 4  # scale down to rougly match the real275 dataset scale
        sizes = sizes | cate_sizes

    np.save(join(out_path, "mean_syn_scales.npy"), mean_scales)

    mean_scales = {}
    # get the mean scale from the real objects
    with open(join(mesh_path, "real_train.pkl"), "rb") as f:
        real_models = cPickle.load(f)
    for cate in classes:
        mean_scale = np.array([np.linalg.norm(2 * np.max(load_obj(join(mesh_path, "real_train", k + ".obj"))[0],
                                axis=0)) for k, v in real_models.items()
                                if k.startswith(cate)]).mean()
        mean_scales[cate] = mean_scale
    np.save(join(out_path, "mean_real_scales.npy"), mean_scales)

def create_prior_meshes_h6d(cfg, n_vertices=1000):
    classes = cfg.dataset.classes
    mesh_path = cfg.dataset.paths.cad_path
    out_path = cfg.dataset.paths.mesh_path
    mean_scales = {}
    for cate in tqdm(classes):

        mesh_dir = join(mesh_path, cate, "*.txt")

        obj_paths = glob.glob(mesh_dir)
        _, mean_scale, xvert, xface = generate_mean_mesh(obj_paths, n_vertices=n_vertices, save_sizes=False)
        save_off(join(out_path, f"{cate}.off"), xvert, xface)
        mean_scales[cate] = mean_scale

    np.save(join(out_path, "mean_scales.npy"), mean_scales)
