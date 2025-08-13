import os
import math
import numpy as np
import _pickle as cPickle
import json
from torchvision.transforms import v2 as tf
import glob
from PIL import Image

from .dataset import PoseDataset, DatasetCfg

# CAMERA training set with 249127 images, val set 4662, no test set
# Real training set with 4318 images, test set with 2754

sample_keys = [
    "img",
    "label",  # list[int] (n)
    "R",  # list[Tensor] (n,3,3)
    "T",  # list[Tensor] (n,3,3)
    "num_object",  # int
    "K",  # Tensor [3,3]
    "sizes",  # list[Tensor] (n,3)
]


class H6D(PoseDataset):
    def __init__(self, cfg: DatasetCfg, for_test=False) -> None:
        super().__init__(cfg, for_test)
        self.setup()
    def setup(self):
        mode = "test" if self.cfg.for_test else "train"
        self.mode = mode
        self.data_dir = self.cfg.paths.data_path


        img_list = []
        if self.cfg.for_test:
            scenes = glob.glob(os.path.join(self.data_dir, "test_scene*/"))
        else:
            scenes = glob.glob(os.path.join(self.data_dir, "scene*/"))
        for scene in scenes:
            scene_list = os.listdir(os.path.join(scene, "rgb"))
            img_list += [scene+"rgb/"+img for img in scene_list]

        self.img_list = img_list
        self.length = len(self.img_list)

        self.sym_ids = [0, 2, 5, 9]  # 0-indexed
        self.mean_scales = np.load(os.path.join(self.cfg.paths.mesh_path, "mean_scales.npy"), allow_pickle=True)[()]



    def get_mean_scale(self, labels):
        return np.array([self.mean_scales[self.classes[int(label)]] for label in labels])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.img_list[index])
        rgb = Image.open(img_path)

        anno_path = img_path.replace("rgb", "labels").replace(".png", "_label.pkl")
        sample = dict.fromkeys(sample_keys)

        with open(anno_path, "rb") as f:
            gts = cPickle.load(f)

        sample["K"] = np.loadtxt(os.path.join(img_path.split('rgb')[0], 'intrinsics.txt')).reshape(3,3).astype(np.float32)
        sample["img"] = rgb
        sample["name_img"] = img_path
        # Classes in Housecat are not sorted alphabetically, so we convert to 0-indexed
        sample["label"] = np.array([int(self.cfg.classes.index(cls.split("-")[0])) for cls in gts["model_list"]])
        sample["num_object"] = len(sample["label"])
        sample["R"] = gts["rotations"].astype(np.float32)
        self._map_to_canonical_rotation(sample)
        sample["R"] = sample["R"].transpose(0, 2, 1) @ np.array(
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        sample["T"] = gts["translations"].astype(np.float32)
        sample["T"][..., :2] *= -1

        scales = self.mean_scales
        sample["mean_scale"] = np.array([scales[self.classes[int(lbl)]] for lbl in sample["label"]])

        sample["size"] = gts["gt_scales"].astype(np.float32)

        sample["scale"] = gts["scales"].astype(np.float32)
        sample["size"] /= sample["scale"][:,None]

        # only test data has gt handle visibility
        if "handle_visibility" in gts:
            sample["handle_visibility"] = gts["handle_visibility"]
        else:
            sample["handle_visibility"] = np.ones(sample["num_object"])

        sample = self.prepare_im(sample)

        return sample

    def _map_to_canonical_rotation(self, sample, real=False):
        for instance_id in range(sample["num_object"]):

            handle_mug = False
            if real and (sample['label'][instance_id] == 5) and (not self.cfg.for_test):
                handle_tmp_path = sample["name_img"].split('/')
                scene_label = handle_tmp_path[-2] + '_res'
                img_id = int(handle_tmp_path[-1])
                handle_mug = self.mug_info[scene_label][img_id] == 0
            if handle_mug or (sample["label"][instance_id] in self.sym_ids):
                rotation = sample["R"][instance_id]

                # assume continuous axis rotation symmetry
                theta_x = rotation[0, 0] + rotation[2, 2]
                theta_y = rotation[0, 2] - rotation[2, 0]
                r_norm = math.sqrt(theta_x**2 + theta_y**2)
                if r_norm == 0:
                    continue
                s_map = np.array(
                    [
                        [theta_x / r_norm, 0.0, -theta_y / r_norm],
                        [0.0, 1.0, 0.0],
                        [theta_y / r_norm, 0.0, theta_x / r_norm],
                    ]
                )

                if np.any(np.isnan(s_map)) == True:
                    print("invalid rotation matrix")

                canonical_R = rotation @ s_map
                if handle_mug and sample["label"][instance_id] == 5:
                    canonical_R = self.rotate_y_axis(canonical_R)

                sample["R"][instance_id] = canonical_R

