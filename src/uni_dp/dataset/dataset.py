from copy import deepcopy
from dataclasses import dataclass

import torch
import numpy as np
from omegaconf import MISSING, DictConfig, ListConfig
from torch.utils.data import Dataset
import torchvision.transforms.v2 as tf
from .transforms import (
    Normalize,
    ToTensor,
    ToDtype,
    ColorJitter,
    RandomRGBNoise
)


@dataclass
class DatasetCfg:
    name: str
    image_shape: ListConfig[int]
    paths: DictConfig
    classes: ListConfig[str]
    for_test: bool = MISSING
    source: str = MISSING
    syn_ratio: int = MISSING


class PoseDataset(Dataset):
    cfg: DatasetCfg
    file_list: list = []
    label_list: list = []

    def __init__(self, cfg: DatasetCfg, for_test=False) -> None:
        self.cfg = deepcopy(cfg)
        self.cfg.for_test = for_test
        if self.cfg.for_test:
            transform = [ToTensor(), ToDtype(), Normalize()]
        else:
            transform = [ColorJitter(), RandomRGBNoise(), ToTensor(), ToDtype(), Normalize()]
        self.prepare_im = tf.Compose(transform)

    def prep_epoch(self, epoch: int) -> None:
        pass

    def __getitem__(self, item: int) -> dict:
        raise NotImplementedError("Must implement __getitem__")

    def __len__(self) -> int:
        raise NotImplementedError("Must implement __len__")

    @property
    def classes(self) -> ListConfig[str]:
        return self.cfg.classes

    def rotate_y_axis(self, canonical_R):
        theta = np.pi / 2 # 90 degrees
        R_y = np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ],
            dtype=np.float32,
        )
        return canonical_R @ R_y

class DummyDataset(PoseDataset):
    def __init__(self, cfg: DatasetCfg, for_test=False) -> None:
        super().__init__(cfg, for_test)

    def __len__(self) -> int:
        return 100
    def __getitem__(self, item: int) -> dict:
        rgb = np.random.randn(self.cfg.image_shape[0], self.cfg.image_shape[1], 3)

        sample = {}

        sample["K"] = np.array(
            [
                [1.0, 0.0, self.cfg.image_shape[1] / 2],
                [0.0, 1.0, self.cfg.image_shape[0] / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        sample["img"] = rgb
        sample["label"] = np.arange(len(self.cfg.classes))  # convert to 0-indexed
        sample["num_object"] = len(sample["label"])
        sample["R"] = np.stack([np.eye(3, 3, dtype=np.float32) for _ in range(sample["num_object"])], axis=0)
        sample["T"] = np.zeros((sample["num_object"], 3),  dtype=np.float32)

        scales = np.ones_like(self.cfg.classes, dtype=np.float32)
        sample["mean_scale"] = np.array([scales[lbl] for lbl in sample["label"]], dtype=np.float32)

        sample["size"] = np.random.randn(sample["num_object"], 3).astype(np.float32)

        sample["scale"] = np.linalg.norm(sample["size"], axis=-1)
        sample["size"] /= sample["scale"][:, None]

        sample["handle_visibility"] = np.ones(sample["num_object"])

        sample = self.prepare_im(sample)

        return sample

def list_collate_fn(batch):
    return [{k: torch.tensor(v) if isinstance(v, np.ndarray) else v for k, v in t.items()} for t in batch]
