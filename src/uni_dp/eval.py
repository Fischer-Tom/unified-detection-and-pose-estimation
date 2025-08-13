import argparse
from omegaconf import (DictConfig, OmegaConf)
import os
from os.path import join, exists

from torch import load
from models.trainer import Trainer
from tools import set_seeds

def main(args):
    try:
        scale_net_ckpt = load(join(args.load_path, args.scale_net_path)) if args.scale_net_path else None

        chkpt = load(join(args.load_path, args.model_path), map_location='cpu', weights_only=False)
        cfg = chkpt['cfg']
        cfg.checkpointing.result_dir = join(args.load_path, f'results_{args.model_path.split(".")[0]}')
       
    except Exception as e:
        print(f"Error loading checkpoint {args.model_path} at path {args.load_path} with Exception\n: {e}")
        return
    if not exists(cfg.dataset.paths.mesh_path):
        try:
            os.makedirs(cfg.dataset.paths.mesh_path, exist_ok=True)
            if cfg.dataset.name.lower() == "h6d":
                from create_mesh import create_prior_meshes_h6d as create_prior_meshes
            elif cfg.dataset.name.lower() == "real275":
                from create_mesh import create_prior_meshes_real275 as create_prior_meshes
            else:
                raise ValueError(f"Dataset {cfg.dataset.name} not supported for mesh creation.")
            create_prior_meshes(cfg)
        except Exception as e:
            print(f"Failed to create prior meshes: {e}")
            return

    cfg.ddp.distributed = False
    if args.similarity_thr > 0:
        cfg.params.inference.similarity_thr = args.similarity_thr

    os.makedirs(cfg.checkpointing.result_dir, exist_ok=True)
    print(f"Evaluating checkpoint {args.model_path} with configuration: \n",  OmegaConf.to_yaml(cfg))

    model_path = join(args.load_path, args.model_path)
    print(f"Loading model from {model_path}")
    trainer = Trainer(cfg, train_mode=False)
    trainer.load_checkpoint(chkpt,scale_net_ckpt, resume_training=False)

    set_seeds(cfg.seed)
    trainer.validate_model()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Model Evaluation")
    args.add_argument("--load_path", type=str, default="weights")
    args.add_argument("--model_path", type=str, default="Real.pt")
    args.add_argument("--scale_net_path", type=str, default="scale_net.pth")
    args.add_argument("--similarity_thr", type=float, default=-1.0)

    args = args.parse_args()

    main(args)
