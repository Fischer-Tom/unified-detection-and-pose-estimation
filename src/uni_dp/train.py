import os, pathlib
def _is_rank0_before_ddp() -> bool:
    return os.environ.get("RANK", "0") == "0"
if not _is_rank0_before_ddp():
    os.environ["HYDRA_JOB_CHDIR"] = "false"
    os.environ["HYDRA_OUTPUT_SUBDIR"] = ""
    os.environ["HYDRA_RUN_DIR"] = str(pathlib.Path.cwd())
    os.environ.setdefault("WANDB_MODE", "disabled")

from typing import Optional
import hydra
from omegaconf import (DictConfig, OmegaConf)
import wandb
import os
import torch
from os.path import join, exists
from models.trainer import Trainer
import torch.distributed as dist
from tools.ddp_utils import init_distributed_mode, is_main_process, dprint
from tools import set_seeds

def find_latest_checkpoint(checkpoint_dir: str, checkpoint_prefix: str = "UniDP_") -> Optional[str]:
    """
    Find the latest checkpoint file in the given directory based on iteration number.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        checkpoint_prefix: Prefix pattern for checkpoint files (unused but kept for compatibility)
        
    Returns:
        Path to the latest checkpoint file, or None if no checkpoints found
        
    Raises:
        ValueError: If checkpoint filenames don't match expected format
    """
    try:
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        if not checkpoints:
            return None
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
        return join(checkpoint_dir, latest_checkpoint)
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid checkpoint filename format in {checkpoint_dir}") from e

def setup_distributed_training(cfg: DictConfig) -> DictConfig:
    """
    Automatically configure distributed training based on environment and available GPUs.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Updated configuration with proper DDP settings
    """
    # Check if we're in a torchrun environment
    torchrun_detected = "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ
    
    if torchrun_detected:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dprint(f"Torchrun detected: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        
        # Auto-configure DDP settings
        cfg.ddp.distributed = world_size > 1
        cfg.ddp.world_size = world_size
        cfg.ddp.rank = rank
        cfg.ddp.gpu = local_rank
        
        if cfg.ddp.distributed:
            dprint(f"Multi-GPU training enabled: using {world_size} GPUs")
        else:
            print("Single GPU training (world_size=1)")
            
    else:
        # Check for available GPUs if not using torchrun
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"Detected {gpu_count} GPU(s), torchrun not detected")
        
        if gpu_count <= 1:
            cfg.ddp.distributed = False
            cfg.ddp.world_size = 1
            cfg.ddp.gpu = 0 if gpu_count == 1 else None
            print("Single GPU/CPU training mode")
        else:
            print(f"Warning: {gpu_count} GPUs detected but torchrun not used. Use train.sh for multi-GPU training.")
            cfg.ddp.distributed = False
            cfg.ddp.world_size = 1
            cfg.ddp.gpu = 0
    
    return cfg


@hydra.main(version_base="1.3", config_path="../../conf", config_name="main")
def main(cfg: DictConfig) -> None:
    try:
        log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        config_path = join(log_dir, "config.yaml")
        resume_training = os.path.exists(config_path)

        if resume_training:
            dprint(f"Resuming from checkpoint {log_dir}")
            cfg = OmegaConf.load(config_path)
        else:
            cfg.checkpointing.log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            cfg.checkpointing.result_dir = join(cfg.checkpointing.log_dir, 'results')

        cfg = setup_distributed_training(cfg)

        # Only init wandb on main (optional)
        run = None
        if is_main_process():
            try:
                run = wandb.init(
                    project=cfg.wandb.project,
                    notes=cfg.wandb.notes,
                    config=dict(cfg.params),
                    mode=cfg.wandb.mode,
                    dir=cfg.checkpointing.log_dir,
                )
                dprint("Wandb initialized successfully")
            except Exception as e:
                dprint(f"Warning: Failed to initialize wandb: {e}")

        if hasattr(cfg, 'ddp') and cfg.ddp.distributed:
            try:
                distributed = init_distributed_mode(cfg.ddp)
                if not distributed:
                    cfg.ddp.distributed = False
                    print("Distributed training initialization failed, falling back to single GPU")
                else:
                    dprint(f"Distributed training initialized successfully on {cfg.ddp.world_size} GPUs")
            except Exception as e:
                print(f"Warning: Failed to initialize distributed training: {e}")
                print("Falling back to single GPU training")
                cfg.ddp.distributed = False

        # Mesh creation (rank 0 only) + safe barrier
        if not exists(cfg.dataset.paths.mesh_path):
            dprint("Mesh directory not found. Creating meshes ...")
            if is_main_process():
                try:
                    os.makedirs(cfg.dataset.paths.mesh_path, exist_ok=True)
                    if cfg.dataset.name.lower() == "h6d":
                        from create_mesh import create_prior_meshes_h6d as create_prior_meshes
                    elif cfg.dataset.name.lower() == "real275":
                        from create_mesh import create_prior_meshes_real275 as create_prior_meshes
                    else:
                        raise ValueError(f"Dataset {cfg.dataset.name} not supported for mesh creation.")
                    create_prior_meshes(cfg)
                    print("Meshes created successfully")
                except Exception as e:
                    raise RuntimeError(f"Failed to create meshes: {e}") from e

        if dist.is_available() and dist.is_initialized():
            dist.barrier()


        # Set random seeds
        try:
            set_seeds(cfg.seed)
            dprint(f"Random seeds set to {cfg.seed}")
        except Exception as e:
            dprint(f"Warning: Failed to set seeds: {e}")

        # Initialize trainer
        trainer = Trainer(cfg)
        dprint("Trainer initialized")

        # Resume from checkpoint if needed
        if resume_training:
            try:
                chkpt = find_latest_checkpoint(cfg.checkpointing.log_dir)
                if chkpt:
                    trainer.load_checkpoint(chkpt, resume_training=True)
                else:
                    dprint("Warning: No checkpoint found for resuming")
            except Exception as e:
                dprint(f"Error: Failed to resume from checkpoint: {e}")
                raise
        
        # Save current config
        try:
            OmegaConf.save(cfg, join(cfg.checkpointing.log_dir, 'config.yaml'))
        except Exception as e:
            dprint(f"Warning: Failed to save config: {e}")

        # Start training - detailed logging handled by trainer
        trainer.train_model()

    except Exception as e:
        dprint(f"Error: Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
