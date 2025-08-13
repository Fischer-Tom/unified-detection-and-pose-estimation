import numpy as np
import torch
import _pickle as cPickle
from contextlib import nullcontext
import signal

import os
import sys
from os.path import join, exists
from typing import TYPE_CHECKING, List
import threading
from torch import tensor as Tensor
import torch.distributed as dist

from src.uni_dp.libs import Logger, RenderEngine
from src.uni_dp.tools import (parse_meshes, construct_optimizer,
                              reduce_dict, save_on_master, evaluator, convert_RTs_from_cv2_to_py3d, is_main_process)
from src.uni_dp.dataset import collate_fn
from .scalenet import ScaleNet

from .mesh_memory import NeuralMesh
from .uni_posedet import UniDP

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from omegaconf import DictConfig
    from torch.nn import Module


class Trainer:
    current_pad_index: Tensor
    n_classes: int = None
    current_task_id: int = 0
    t_iter: int = 0
    pad_index: Tensor = None

    def __init__(
        self,
        cfg: "DictConfig",
        train_mode: bool = True,
    ) -> None:
        self.model = None
        self.model_without_ddp = None
        self.optimizer = None
        self.lr_scheduler = None
        self.renderer = None
        self.dataset = None
        self.test_dataset = None
        self.cfg = cfg
        self.logger = Logger(
            total_iters=cfg.optimizer.train_iters,
            log_dir=cfg.checkpointing.log_dir,
            experiment_name=f"{cfg.dataset.name}",
            save_frequency=100,  # Log metrics every 100 iterations
            metrics_window_size=200,  # Track last 200 iterations for averages
        )
        self._setup_training(train_mode=train_mode)
        self._setup_data(train_mode=train_mode)

        # Optional: If the compute cluster sends SIGTERM to stop training,
        self._term_event = threading.Event()
        self._usr1_event = threading.Event()
        self._install_signal_handlers()

        
        # Log system info and hyperparameters
        if train_mode:
            self.logger.log_system_info()
            self.logger.log_hyperparameters(dict(cfg))

    def _install_signal_handlers(self):
        """Register SLURM-friendly signal handlers (rank-agnostic)."""
        # Handle SIGUSR1 (SLURM warning signal)
        signal.signal(signal.SIGUSR1, self._on_usr1)
        # Handle SIGTERM (normal termination signal)
        signal.signal(signal.SIGTERM, self._on_term)
        # Handle SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, self._on_term)
    def _on_usr1(self, signum, frame):
        self._usr1_event.set()
        self.logger.info("[Signal] Received SIGUSR1, saving checkpoint and terminating gracefully.")

    def _on_term(self, signum, frame):
        self._term_event.set()
        signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        self.logger.info(f"[Signal] Received {signal_name}, terminating gracefully.")

    def _should_stop(self):
        # share stop intent across ranks
        t = torch.tensor([1 if (self._usr1_event.is_set() or self._term_event.is_set()) else 0],
                         device='cuda' if torch.cuda.is_available() else 'cpu')
        if hasattr(self.cfg, 'ddp') and self.cfg.ddp.distributed:
            # Use distributed all_reduce to check if any rank has set the stop event
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t.item() > 0

    def save_checkpoint(self) -> None:
        """Save model checkpoint including model state, optimizer state, and scheduler state."""
        try:
            save_path = f"{str(self.model_without_ddp)}_{self.t_iter}.pt"
            checkpoint_path = join(self.cfg.checkpointing.log_dir, save_path)
            
            self.logger.info(f"Saving model checkpoint: {checkpoint_path}")
            
            # Ensure checkpoint directory exists
            os.makedirs(self.cfg.checkpointing.log_dir, exist_ok=True)
            
            checkpoint = {
                "model": self.model_without_ddp.state_dict(),
                "optimizer": self.optimizer.state_dict() if self.optimizer else None,
                "scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                "epoch": self.t_iter,
                "cfg": self.cfg,
            }
            if self.scaler is not None:
                checkpoint["scaler"] = self.scaler.state_dict()
                
            save_on_master(checkpoint, checkpoint_path)
            self.logger.info(f"Successfully saved checkpoint: {checkpoint_path}")
            
        except Exception as e:
            error_msg = f"Failed to save checkpoint at iteration {self.t_iter}: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def load_checkpoint(self, chkpt: str | dict, scale_net_ckpt: str | dict = None, resume_training: bool = False) -> None:
        """
        Load checkpoint from file path or dictionary.
        
        Args:
            chkpt: Path to checkpoint file or checkpoint dictionary
            resume_training: If True, load optimizer and scheduler states for training continuation
            
        Raises:
            ValueError: If checkpoint format is invalid
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        try:
            if isinstance(chkpt, str):
                if not os.path.exists(chkpt):
                    raise FileNotFoundError(f"Checkpoint file not found: {chkpt}")
                self.logger.info(f"Loading checkpoint from: {chkpt}")
                checkpoint = torch.load(chkpt, weights_only=False, map_location='cpu')
            elif isinstance(chkpt, dict):
                checkpoint = chkpt
            else:
                raise ValueError("Checkpoint must be a path to a file or a dictionary")
                
            # Validate checkpoint contents
            if "model" not in checkpoint:
                raise ValueError("Checkpoint missing 'model' key")
                
            # Load model state
            self.model_without_ddp.load_state_dict(checkpoint["model"])
            
            self.logger.info("Successfully loaded model state")
            
            if resume_training:
                if self.optimizer is None:
                    raise RuntimeError("Optimizer must be initialized to resume training")
                    
                if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                    self.logger.info("Loaded optimizer state")
                else:
                    self.logger.warning("No optimizer state found in checkpoint")
                    
                if "scheduler" in checkpoint and checkpoint["scheduler"] is not None and self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
                    self.logger.info("Loaded scheduler state")
                    
                if "scaler" in checkpoint and checkpoint["scaler"] is not None and self.scaler is not None:
                    self.scaler.load_state_dict(checkpoint["scaler"])
                    self.logger.info("Loaded scaler state")
                    
                self.t_iter = checkpoint.get("epoch", 0)
                self.logger.resume_training(self.t_iter)
                self.logger.info(f"Resumed training from iteration {self.t_iter}")
            else:
                self.model_without_ddp.normalize_memory()
                self.logger.info("Loaded checkpoint for inference (normalized memory)")
                
        except Exception as e:
            error_msg = f"Failed to load checkpoint: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        if scale_net_ckpt is not None:
            self.model_without_ddp.scale_net = ScaleNet().cuda()
            self.model_without_ddp.scale_net.load_state(scale_net_ckpt)
        else:
            self.model_without_ddp.scale_net = None

    def _setup_training(self, train_mode: bool) -> None:
        """
        Initialize model, optimizer, scheduler and distributed training setup.
        
        Args:
            train_mode: Whether to setup for training (includes optimizer) or inference only
        """
        xverts, xfaces = parse_meshes(self.cfg)
        self.cfg.params.mesh.max_n = max([vert.shape[0] for vert in xverts])
        self.renderer = RenderEngine(self.cfg)
        neural_mesh = NeuralMesh(xverts, xfaces, self.cfg.params.mesh)
        self.model = UniDP(
            neural_mesh=neural_mesh,
            cfg=self.cfg.params,
        )
        
        # Move model to GPU with error handling
        try:
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                self.logger.info(f"Model moved to GPU: {torch.cuda.get_device_name()}")
            else:
                self.logger.warning("CUDA not available, using CPU")
                self.model = self.model.cpu()
        except Exception as e:
            self.logger.warning(f"Failed to move model to GPU, using CPU: {e}")
            self.model = self.model.cpu()
            
        self.model_without_ddp = self.model
        if train_mode:
            if hasattr(self.cfg, 'ddp') and self.cfg.ddp.distributed:
                try:
                    self.model = torch.nn.parallel.DistributedDataParallel(
                        self.model, 
                        device_ids=[self.cfg.ddp.gpu] if torch.cuda.is_available() else None,
                        find_unused_parameters=True
                    )
                    self.model_without_ddp = self.model.module
                    self.logger.info("DDP wrapper applied successfully")
                except Exception as e:
                    self.logger.warning(f"Failed to apply DDP, using single GPU: {e}")
                    self.model_without_ddp = self.model
                    
            self.optimizer, self.lr_scheduler = construct_optimizer(
                self.model, self.cfg.optimizer
            )

            use_fp16 = (self.cfg.params.mesh.precision == "half")
            use_bf16 = (self.cfg.params.mesh.precision == "bf16")
            assert not (use_fp16 and use_bf16), "Choose either half or bf16, not both."
            # Setup gradient scaler with CUDA check
            self.scaler = torch.cuda.amp.GradScaler() if (torch.cuda.is_available() and use_fp16) else None
        self._set_current_pad_index(xverts)

    def _setup_data(self, train_mode: bool = False) -> None:
        """
        Initialize datasets for training and testing.
        
        Args:
            train_mode: Whether to setup training dataset
            
        Raises:
            NotImplementedError: If dataset type is not supported
            RuntimeError: If dataset initialization fails
        """
        try:
            # Validate dataset configuration
            if not hasattr(self.cfg, 'dataset') or not hasattr(self.cfg.dataset, 'name'):
                raise ValueError("Missing dataset configuration")
                
            dataset_name = self.cfg.dataset.name
            self.logger.info(f"Setting up dataset: {dataset_name}")
            
            if dataset_name.lower() == "real275":
                from src.uni_dp.dataset import Real275 as ds
            elif dataset_name.lower() == "h6d":
                from src.uni_dp.dataset import H6D as ds
            elif dataset_name.lower() == "dummy":
                from src.uni_dp.dataset import DummyDataset as ds
            else:
                raise NotImplementedError(f"Dataset '{dataset_name}' not implemented. "
                                        f"Supported datasets: Real275, H6D, Dummy")
            
            # Initialize datasets with error handling
            if train_mode:
                try:
                    self.dataset = ds(self.cfg.dataset)
                    self.logger.info(f"Training dataset initialized with {len(self.dataset)} samples")
                except Exception as e:
                    raise RuntimeError(f"Failed to initialize training dataset: {e}") from e
                    
            try:
                self.test_dataset = ds(self.cfg.dataset, for_test=True)
                self.logger.info(f"Test dataset initialized with {len(self.test_dataset)} samples")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize test dataset: {e}") from e
                
        except Exception as e:
            self.logger.error(f"Failed to setup datasets: {e}")
            raise

    def _get_data_loaders(self, epoch: int = 0, train_mode: bool = True):
        if train_mode:

            def _ignore_usr1_in_workers(_):
                signal.signal(signal.SIGUSR1, signal.SIG_IGN)

            train_sampler = None
            self.dataset.prep_epoch(epoch)
            if hasattr(self.cfg, 'ddp') and self.cfg.ddp.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=True,
                                                                                drop_last=True)
            loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.cfg.data_loader.train.batch_size,
                shuffle=train_sampler is None,
                num_workers=self.cfg.data_loader.train.num_workers,
                collate_fn=collate_fn,
                sampler=train_sampler,
                persistent_workers=False,
                pin_memory=True,
                drop_last=True,
                worker_init_fn=_ignore_usr1_in_workers
            )
        else:
            loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.cfg.data_loader.test.batch_size,
                shuffle=True,
                num_workers=self.cfg.data_loader.test.num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
                drop_last=False,
                generator=torch.Generator().manual_seed(self.cfg.seed),
            )

        return loader

    def train_model(self) -> None:
        """
        Execute the main training loop until target iterations are reached.
        
        Raises:
            RuntimeError: If training fails due to critical errors
        """
        try:
            with self.logger:
                self.logger.info(f"Starting training for {self.cfg.optimizer.train_iters} iterations")
                epoch = -1
                while self.t_iter < self.cfg.optimizer.train_iters:
                    try:
                        epoch += 1
                        # Get data loader with error handling
                        train_loader = self._get_data_loaders(epoch, train_mode=True)
                        self.model.train()
                        # Set epoch for distributed training
                        if hasattr(self.cfg, 'ddp') and self.cfg.ddp.distributed:
                            train_loader.sampler.set_epoch(epoch)

                        # Training iteration loop
                        for i, sample in enumerate(train_loader):
                            try:
                                self.model_without_ddp.current_sample = i
                                self.t_iter += 1
                                
                                # Execute training step with error handling
                                self.train_step(sample)

                                # Save checkpoint periodically
                                if (self.t_iter + 1) % self.cfg.checkpointing.save_every == 0:
                                    try:
                                        self.save_checkpoint()
                                    except Exception as e:
                                        self.logger.warning(f"Failed to save checkpoint at iteration {self.t_iter}: {e}")
                                        
                                # Check if training is complete
                                if self.t_iter >= self.cfg.optimizer.train_iters:
                                    self.logger.info(f"Reached target iterations: {self.cfg.optimizer.train_iters}")
                                    self._graceful_terminate()
                                    self.logger.info("Training completed successfully")
                                    return

                                # Check if any termination signal is received
                                if self._should_stop():
                                    self._graceful_terminate()
                                    return

                            except RuntimeError as e:
                                if "out of memory" in str(e).lower():
                                    self.logger.warning(f"OOM error at iteration {self.t_iter}, skipping batch")
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    continue
                                else:
                                    self.logger.error(f"Critical error during training step: {e}")
                                    raise
                            except Exception as e:
                                self.logger.error(f"Unexpected error at iteration {self.t_iter}: {e}")
                                raise RuntimeError(f"Training failed at iteration {self.t_iter}") from e
                                
                    except Exception as e:
                        self.logger.error(f"Error setting up data loader or model: {e}")
                        raise RuntimeError("Failed to setup training components") from e
                        

        except Exception as e:
            self.logger.error(f"Training failed at iteration {self.t_iter}: {e}")
            raise

    def train_step(self, sample: List) -> None:
        """
        Execute a single training step with forward pass, loss calculation, and parameter updates.
        
        Args:
            sample: Batch of training samples
            
        Raises:
            RuntimeError: If training step fails due to memory or computation errors
        """
        try:
            # Prepare annotations with error handling
            try:
                annos = self._prepare_annotations(sample)
            except Exception as e:
                raise RuntimeError(f"Failed to prepare annotations: {e}") from e
            
            # Forward pass with automatic mixed precision
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            use_fp16 = (self.cfg.params.mesh.precision == "half")
            use_bf16 = (self.cfg.params.mesh.precision == "bf16")
            if use_fp16:
                ctx = torch.cuda.amp.autocast(dtype=torch.float16)
            elif use_bf16:
                ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16)
            else:
                # no autocast
                ctx = nullcontext()
            try:
                with ctx:
                    out_dict = self.model(*annos)
            except torch.cuda.OutOfMemoryError:
                self.logger.warning("CUDA out of memory during forward pass - clearing cache")
                torch.cuda.empty_cache()
                raise RuntimeError("GPU out of memory during forward pass")
            except Exception as e:
                raise RuntimeError(f"Forward pass failed: {e}") from e
                
            # Calculate total loss
            try:
                loss = sum(out_dict[k] for k in out_dict.keys() if "loss" in k)
                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError(f"Invalid loss value: {loss}")
            except Exception as e:
                raise RuntimeError(f"Loss calculation failed: {e}") from e
            
            # Backward pass and optimization
            try:
                self.optimizer.zero_grad()
                
                if self.scaler is not None:
                    # Mixed precision training
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Full precision training
                    loss.backward()
                    self.optimizer.step()
                    
                # Update learning rate
                self.lr_scheduler.step()
                
            except torch.cuda.OutOfMemoryError:
                self.logger.warning("CUDA out of memory during backward pass - clearing cache")
                torch.cuda.empty_cache()
                raise RuntimeError("GPU out of memory during backward pass")
            except Exception as e:
                raise RuntimeError(f"Backward pass or optimization failed: {e}") from e
            
            # Reduce metrics across processes and log
            try:
                reduced_dict = reduce_dict(out_dict, reduce=True, to_gather=["loss", "mask_loss"])
                
                # Get learning rate for logging
                current_lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else 0.0
                
                self.logger.step(
                    iteration=self.t_iter, 
                    metrics=reduced_dict,
                    learning_rate=current_lr
                )
            except Exception as e:
                self.logger.warning(f"Failed to reduce metrics or log: {e}")
                # Continue training even if logging fails
                
        except Exception as e:
            self.logger.error(f"Training step failed at iteration {self.t_iter}: {e}")
            raise

    @torch.inference_mode()
    def validate_model(self) -> None:
        """
        Validate model on test dataset with structured logging and progress tracking.
        
        Raises:
            RuntimeError: If validation fails due to critical errors
        """
        try:
            # Setup validation parameters
            if self.cfg.params.mesh.precision == "half":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
            
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            test_loader = self._get_data_loaders(train_mode=False)
            self.model.eval()
            
            # Ensure results directory exists
            os.makedirs(self.cfg.checkpointing.result_dir, exist_ok=True)
            
            # Start validation with progress tracking
            self.logger.start_validation(len(test_loader))
            
            img_count = 0
            inst_count = 0
            
            try:
                for batch_idx, sample in enumerate(test_loader):
                    if batch_idx == 10:
                        break
                    try:
                        if self._should_stop():
                            return
                        # Move data to device and run inference
                        sample = self._to_device(sample)
                        with torch.amp.autocast(device_type=device_type, dtype=dtype):
                            poses = self.model_without_ddp.inference(sample)

                        # Process each prediction-ground truth pair
                        for pred, gt in zip(poses, sample):
                            img_count += 1
                            inst_count += len(gt["label"])
                            
                            # Extract predictions
                            output = {}
                            est_R, est_T, est_size, est_label,est_scale = pred["estimations"]
                            
                            # Process estimations
                            if len(est_label) > 0:
                                est_R, est_T = convert_RTs_from_cv2_to_py3d(
                                    est_R, est_T, gt["K"],
                                    torch.tensor(gt["img"].shape[1:]) / self.cfg.params.extractor.downsample_rate
                                )
                                est_R = np.stack(est_R, axis=0)
                                est_T = np.stack(est_T, axis=0)
                                est_size = np.stack(est_size, axis=0)
                                est_label = torch.stack(est_label, axis=0).cpu().numpy() + 1
                            else:
                                est_R = np.zeros((0, 3, 3), dtype=np.float32)
                                est_T = np.zeros((0, 3), dtype=np.float32)
                                est_scale = np.zeros((0,), dtype=np.float32)
                                est_size = np.zeros((0, 3), dtype=np.float32)
                                est_label = np.zeros((0,), dtype=np.float32)

                            # Prepare pose matrices
                            est_poses = np.eye(4, dtype=np.float32)[None,...].repeat(len(est_label), axis=0)
                            gt_poses = np.eye(4, dtype=np.float32)[None,...].repeat(len(gt["label"]), axis=0)

                            # Process ground truth poses
                            gt_poses[:, :3, :3] = np.transpose(
                                (gt["R"].cpu().numpy()*np.stack(gt["scale"].cpu().numpy())[:, None, None]
                                 @ np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32,)),
                                [0,2,1])
                            gt_poses[:, :3, 3] = gt["T"].cpu().numpy()
                            gt_poses[:, :2, 3] *= -1

                            # Process estimated poses
                            est_poses[:, :3, :3] = np.transpose(
                                (est_R *est_scale[:, None, None]
                                 @ np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32, )),
                                [0, 2, 1])
                            est_poses[:, :3, 3] = est_T * est_scale[:, None]
                            est_poses[:, :2, 3] *= -1

                            # Build output dictionary
                            output["gt_class_ids"] = gt["label"].cpu().numpy() + 1
                            output["gt_bboxes"] = np.ones((len(gt_poses), 4))
                            output["gt_RTs"] = gt_poses
                            output["gt_scales"] = gt["size"].cpu().numpy()
                            if "handle_visibility" in gt:
                                output["gt_handle_visibility"] = gt["handle_visibility"].cpu().numpy()
                            else:
                                output["gt_handle_visibility"] = np.ones(len(gt["label"]))

                            output["pred_class_ids"] = est_label
                            output["pred_RTs"] = est_poses
                            output["pred_scales"] = est_size
                            output["pred_bboxes"] = np.ones((len(est_label), 4))
                            output["pred_scores"] = np.ones(len(est_label))

                            # Save results
                            image_short_path = "_".join(gt["name_img"].split("/")[-3:])
                            save_path = join(
                                self.cfg.checkpointing.result_dir,
                                "results_{}.pkl".format(image_short_path),
                            )
                            
                            try:
                                with open(save_path, "wb") as f:
                                    cPickle.dump(output, f)
                            except Exception as e:
                                self.logger.warning(f"Failed to save result for {image_short_path}: {e}")
                                
                        # Update progress
                        self.logger.validation_batch_step(batch_idx, img_count, inst_count)
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing batch {batch_idx}: {e}")
                        continue
                        
            finally:
                # Always finish validation progress tracking
                self.logger.finish_validation()
                
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            raise RuntimeError("Validation failed") from e
        
        # Log final validation summary
        try:
            # Write legacy log file for compatibility
            log_file_path = join(self.cfg.checkpointing.result_dir, "eval_logs.txt")
            write_mode = "a" if exists(log_file_path) else "w"
            
            with open(log_file_path, write_mode) as fw:
                messages = [
                    f"Total images: {img_count}",
                    f"Valid images: {img_count}, Total instances: {inst_count}, Average: {inst_count / img_count:.2f}/image"
                ]
                for msg in messages:
                    self.logger.info(msg)
                    fw.write(msg + "\n")
                    
            # Run evaluation and log results
            sym_classes = [self.cfg.dataset.classes[i] for i in self.test_dataset.sym_ids]
            eval_results = evaluator.run_evaluation(
                self.cfg.checkpointing.result_dir, 
                ["BG"] + list(self.cfg.dataset.classes), 
                sym_classes
            )
            
            # Log validation metrics with structured logging
            validation_metrics = {
                "total_images": img_count,
                "total_instances": inst_count,
                "avg_instances_per_image": inst_count / img_count if img_count > 0 else 0.0
            }
            
            # Add evaluation results if available
            if eval_results and isinstance(eval_results, dict):
                validation_metrics.update(eval_results)
                
            self.logger.validation_step(validation_metrics)
            self.logger.info("Validation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to complete validation summary: {e}")
            raise

    @torch.no_grad()
    def _prepare_annotations(
            self,
            sample: List,
            device="cuda",
    ):
        sample = self._to_device(sample)
        anno_m = [self.renderer(
            self.model_without_ddp.neural_mesh.weight,
            s["R"],
            s["T"],
            s["label"],
            s["size"],
            K=s["K"],
            xverts=self.model_without_ddp.neural_mesh.xverts,
            xfaces=self.model_without_ddp.neural_mesh.xfaces,
            device=torch.device(device),
            scale=s["scale"],
            scale_only=True,
        ) for s in sample]

        anno_i =  [self.renderer(
            self.model_without_ddp.neural_mesh.weight,
            s["R"],
            s["T"],
            s["label"],
            s["size"],
            K=s["K"],
            xverts=self.model_without_ddp.neural_mesh.xverts,
            xfaces=self.model_without_ddp.neural_mesh.xfaces,
            device=torch.device(device),
            scale=s["scale"],
            scale_only=False,
        ) for s in sample]

        return sample, anno_m, anno_i, self.pad_index

    def _to_device(self, sample: List, device="cuda"):
        sample = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in sample]
        return sample

    def _set_current_pad_index(self, xverts: List[Tensor]) -> None:
        n_classes = len(xverts)
        pad_idx = torch.zeros(n_classes, self.cfg.params.mesh.max_n, dtype=torch.bool)
        for i in range(n_classes):
            pad_idx[i, xverts[i].shape[0]:] = True
        self.pad_index = pad_idx.to(self.model_without_ddp.device)

    def _graceful_terminate(self):
        """Rank-0 saves checkpoint & syncs, then all ranks exit cleanly."""

        is_ddp = hasattr(self.cfg, 'ddp') and self.cfg.ddp.distributed

        # 1) Rank-0: save checkpoint + optional sync from node-local scratch
        if is_main_process():
            try:
                self.logger.warning(f"Graceful termination requested; saving checkpoint...")
                self.save_checkpoint()
                self.logger.info(f"Checkpoint saved at t_iter={self.t_iter}")
            except Exception as e:
                self.logger.error(f"Failed to save checkpoint during graceful termination: {e}")

        # 2) Let everyone rendezvous so ranks don’t exit early
        if is_ddp:
            try:
                dist.barrier()
            except Exception as e:
                self.logger.error(f"[rank{dist.get_rank()}] barrier during terminate failed: {e}")


        # 3) Exit the process (0 so SLURM sees a clean stop)
        self.logger.info("Exiting after graceful termination.")