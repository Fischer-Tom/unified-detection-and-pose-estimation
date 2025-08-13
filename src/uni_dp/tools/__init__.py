from .mesh_utils import parse_meshes, rescale_mesh, features_to_keypoints, convert_points_py3d_to_cv2, convert_RTs_from_cv2_to_py3d
from .optimizer import construct_optimizer
from .ddp_utils import reduce_dict, save_on_master, is_main_process, broadcast_object, all_gather_tensor_dict

from omegaconf import DictConfig
from torch import manual_seed
from numpy.random import seed as np_seed
from random import seed as random_seed
from cv2 import setRNGSeed

def flatten_config(config: DictConfig) -> DictConfig:
    flat_config = _flatten(config)
    return DictConfig(flat_config)

def _flatten(config: DictConfig, prefix="", flat_config=None):
    if flat_config is None:
        flat_config = {}

    for key, value in config.items():
        new_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, DictConfig):
            _flatten(value, f"{new_key}_", flat_config)
        else:
            flat_config[new_key] = value

    return flat_config
def set_seeds(seed):
    manual_seed(seed)
    np_seed(seed)
    random_seed(seed)
    setRNGSeed(seed)