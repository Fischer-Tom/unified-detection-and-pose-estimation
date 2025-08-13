import os
import math
import numpy as np
import _pickle as cPickle
import random
from PIL import Image
from os.path import join

from torch.distributed import broadcast_object_list

from src.uni_dp.tools.ddp_utils import is_dist_avail_and_initialized, is_main_process
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
    "size",  # list[Tensor] (n,3)
]


class Real275(PoseDataset):
    def __init__(self, cfg: DatasetCfg, for_test=False) -> None:
        super().__init__(cfg, for_test)
        self.setup()
        self.data_list = None

    def setup(self):
        self.data_dir = self.cfg.paths.data_path
        self.camera_intrinsics = [577.5, 577.5, 319.5, 239.5]  # [fx, fy, cx, cy]
        self.real_intrinsics = [591.0125, 590.16775, 322.525, 244.11084]

        img_list_path = [
            "train_list.txt",
            "real_train_list.txt",
            "val_list.txt",
            "real_test_list.txt",
        ]
        if not self.cfg.for_test:
            del img_list_path[2:]
        else:
            del img_list_path[:2]

        if self.cfg.source == "CAMERA":
            del img_list_path[-1]

        elif self.cfg.source == "Real":
            del img_list_path[0]
        else:
            # only use Real to test when source is CAMERA+Real
            if self.cfg.for_test:
                del img_list_path[0]

        self.cate_dict = {
            "bottle": "02876657",
            "bowl": "02880940",
            "camera": "02942699",
            "can": "02946921",
            "laptop": "03642806",
            "mug": "03797390",
        }
        img_list = []
        subset_len = []
        for path in img_list_path:
            img_list += [
                line.rstrip("\n")
                for line in open(join(self.data_dir, path))
            ]
            subset_len.append(len(img_list))
        if len(subset_len) == 2:
            self.subset_len = [subset_len[0], subset_len[1] - subset_len[0]]
        self.img_list = img_list
        self.length = len(self.img_list)

        self.sym_ids = [0, 1, 3]  # 0-indexed]

        self.img_list = img_list
        self.length = len(self.img_list)

        self.mean_real_scales = [0.25276473, 0.24667794, 0.2201454,  0.17641996, 0.5214796,  0.2029803]
        self.mean_syn_scales = [0.24646159, 0.248079, 0.24306236, 0.24762736, 0.24839106, 0.24843372]
        
        with open(join(self.data_dir, "mug_handle.pkl"), "rb") as f:
            self.mug_info = cPickle.load(f)


    def get_mean_scale(self, labels):
        if self.cfg.source == "CAMERA":
            return np.array([self.mean_syn_scales[self.classes[int(label)]] for label in labels])
        else:
            return np.array([self.mean_real_scales[self.classes[int(label)]] for label in labels])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        pth = self.img_list[index] if self.data_list is None else self.img_list[self.data_list[index]]
        img_path = os.path.join(self.data_dir, pth)
        real = "real_" in img_path

        rgb = Image.open(img_path+"_color.png")

        sample = dict.fromkeys(sample_keys)

        with open(img_path + "_label.pkl", "rb") as f:
            gts = cPickle.load(f)

        if real:
            sample["K"] = self._setup_camera(*self.real_intrinsics)
        else:
            sample["K"] = self._setup_camera(*self.camera_intrinsics)
        sample["img"] = rgb
        sample["name_img"] = img_path
        sample["label"] = np.array([cls - 1 for cls in gts["class_ids"]])  # convert to 0-indexed
        sample["num_object"] = len(sample["label"])
        sample["R"] = gts["rotations"]

        # map ambiguous rotation to canonical rotation
        self._map_to_canonical_rotation(sample, real)
        # self.adjust_rotation_matrix(sample)
        sample["R"] = sample["R"].transpose(0, 2, 1) @ np.array(
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )  # [n, 3, 3]

        # [n, 3]
        sample["T"] = gts["translations"]
        sample["T"][..., :2] *= -1

        sample["scale"] = gts["scales"]
        sample["size"] = gts["size"]

        # only test data has gt handle visibility
        if "handle_visibility" in gts:
            sample["handle_visibility"] = gts["handle_visibility"]
        else:
            sample["handle_visibility"] = np.ones(sample["num_object"])
            handle_tmp_path = sample["name_img"].split('/')
            scene_label = handle_tmp_path[-2] + '_res'
            img_id = int(handle_tmp_path[-1])
            handle_mug = False if not real else self.mug_info[scene_label][img_id] == 0
            if handle_mug:
                sample["handle_visibility"][sample["label"] == 5] = 0

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


    def _setup_camera(self, cam_fx, cam_fy, cam_cx, cam_cy):
        return np.array(
            [[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]], dtype=np.float32
        )

    def prep_epoch(self, epoch: int) -> None:
        if self.cfg.source == "CAMERA+Real":
            syn_ratio = self.cfg.syn_ratio
            camera_len = self.subset_len[0]
            real_len = self.subset_len[1]
            camera_indices = list(range(self.subset_len[0]))
            real_indices = list(range(camera_len, camera_len + real_len))
            if not is_dist_avail_and_initialized() or is_main_process():
                rng = random.Random(epoch)
                data_list = (
                    rng.sample(camera_indices, int(syn_ratio * real_len)) + real_indices
                )
                rng.shuffle(data_list)
            else:
                data_list = None

            if is_dist_avail_and_initialized():
                obj_list = [data_list]
                broadcast_object_list(obj_list, src=0)
                data_list = obj_list[0]

            self.data_list = data_list
            self.length = len(self.data_list)
