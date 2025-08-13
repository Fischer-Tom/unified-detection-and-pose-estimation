import sys
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from collections import defaultdict, deque
import threading

import numpy as np
import torch
from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from tqdm import tqdm

from src.uni_dp.tools import is_main_process


class CustomRichHandler(RichHandler):
    """Custom Rich handler that includes experiment name in the display."""
    
    def __init__(self, experiment_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experiment_name = experiment_name
    
    def emit(self, record):
        """Emit a log record with custom formatting."""
        # Add experiment name to the message display
        original_msg = record.getMessage()
        record.msg = f"[bold blue]{self.experiment_name}[/bold blue] | {original_msg}"
        record.args = ()  # Clear args since we've already formatted the message
        super().emit(record)


class LogLevel:
    """Logging levels for structured logging."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class MetricsTracker:
    """Advanced metrics tracking with history and statistics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.global_metrics = defaultdict(list)
        self._lock = threading.Lock()
    
    def update(self, metrics: Dict[str, float]) -> None:
        """Update metrics with thread safety."""
        with self._lock:
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.metrics[key].append(value)
                self.global_metrics[key].append(value)
    
    def get_current_avg(self, key: str) -> float:
        """Get current window average for a metric."""
        with self._lock:
            values = self.metrics.get(key, [])
            return np.mean(values) if values else 0.0
    
    def get_global_stats(self, key: str) -> Dict[str, float]:
        """Get global statistics for a metric."""
        with self._lock:
            values = self.global_metrics.get(key, [])
            if not values:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
            
            values = np.array(values)
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values)
            }
    
    def get_recent_trend(self, key: str, n_points: int = 10) -> str:
        """Get trend direction for recent points."""
        with self._lock:
            values = list(self.metrics.get(key, []))
            if len(values) < n_points:
                return "insufficient_data"
            
            recent = values[-n_points:]
            if len(recent) < 2:
                return "stable"
            
            # Simple trend detection
            slope = (recent[-1] - recent[0]) / len(recent)
            if abs(slope) < 1e-6:
                return "stable"
            return "decreasing" if slope < 0 else "increasing"


class AdvancedLogger:
    """
    Advanced logger with structured logging, metrics tracking, and rich visualization.
    """
    
    def __init__(
        self,
        total_iters: int,
        log_dir: str,
        experiment_name: str = "experiment",
        log_level: int = LogLevel.INFO,
        save_frequency: int = 100,
        metrics_window_size: int = 100,
        enable_rich: bool = None,
        enable_wandb_style: bool = False
    ):
        self.total_iters = total_iters
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_level = log_level
        self.save_frequency = save_frequency
        self.enable_wandb_style = enable_wandb_style
        
        # Determine backend
        if enable_rich is None:
            self.enable_rich = sys.stdout.isatty()
        else:
            self.enable_rich = enable_rich
            
        # Initialize components
        self.metrics_tracker = MetricsTracker(metrics_window_size)
        self.start_time = time.time()
        self.current_iter = 0
        self.init_iter = 0
        
        # Only create directories and files on main process
        if is_main_process():
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        self._setup_file_logging()
        
        # Setup rich components
        if self.enable_rich:
            self._setup_rich_logging()
        else:
            self._setup_simple_logging()
            
        # Training state
        self.is_training = False
        self.progress = None
        self.task_id = None
        self.live_table = None
        
        self.log(LogLevel.INFO, f"Logger initialized for experiment: {experiment_name}")
        self.log(LogLevel.INFO, f"Logging to directory: {self.log_dir}")
    
    def _setup_file_logging(self) -> None:
        """Setup structured file logging."""
        # Main log file paths (all processes need these for path consistency)
        self.main_log_file = self.log_dir / f"{self.experiment_name}.log"
        self.metrics_log_file = self.log_dir / f"{self.experiment_name}_metrics.jsonl"
        
        # Create dedicated logger for this experiment
        self.logger = logging.getLogger(f"UniDP.{self.experiment_name}")
        self.logger.setLevel(self.log_level)
        
        # Remove any existing handlers to avoid duplication
        self.logger.handlers.clear()
        
        # Only create file handler on main process to avoid conflicts
        if is_main_process():
            # Add file handler
            file_handler = logging.FileHandler(self.main_log_file)
            file_handler.setLevel(self.log_level)
            file_formatter = logging.Formatter(
                f'%(asctime)s | %(levelname)8s | {self.experiment_name} | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Prevent propagation to root logger to avoid duplicate messages
        self.logger.propagate = False
    
    def _setup_rich_logging(self) -> None:
        """Setup rich console logging."""
        self.console = Console(file=sys.stderr)
        
        # Add custom rich handler to logger
        rich_handler = CustomRichHandler(
            experiment_name=self.experiment_name,
            console=self.console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False
        )
        rich_handler.setLevel(self.log_level)
        self.logger.addHandler(rich_handler)
    
    def _setup_simple_logging(self) -> None:
        """Setup simple console logging for non-TTY environments."""
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(self.log_level)
        formatter = logging.Formatter('%(asctime)s | %(levelname)8s | %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def log(self, level: int, message: str, **kwargs) -> None:
        """Log a message with specified level."""
        if not is_main_process():
            return
        timestamp = datetime.now().isoformat()
        
        # Add to main log - the Custom Rich handler will add experiment name to display
        if level >= self.log_level:
            if level >= LogLevel.ERROR:
                self.logger.error(message, **kwargs)
            elif level >= LogLevel.WARNING:
                self.logger.warning(message, **kwargs)
            elif level >= LogLevel.INFO:
                self.logger.info(message, **kwargs)
            else:
                self.logger.debug(message, **kwargs)
        
        # Add to structured log
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "iteration": self.current_iter,
            "experiment": self.experiment_name,
            **kwargs
        }
        
        # Only write structured logs on main process
        if is_main_process():
            try:
                with open(self.main_log_file.with_suffix('.jsonl'), 'a') as f:
                    f.write(json.dumps(log_entry, default=self._json_serializer) + '\n')
            except Exception as e:
                print(f"Failed to write structured log: {e}")
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.log(LogLevel.ERROR, message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, **kwargs)
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for objects that aren't natively serializable."""
        try:
            # Handle OmegaConf DictConfig objects
            if hasattr(obj, '_metadata'):  # DictConfig has this attribute
                from omegaconf import OmegaConf
                return OmegaConf.to_container(obj, resolve=True)
            
            # Handle torch tensors
            elif hasattr(obj, 'item'):  # torch.Tensor has this method
                if obj.numel() == 1:
                    return obj.item()
                else:
                    return obj.tolist()
            
            # Handle numpy arrays
            elif hasattr(obj, 'tolist'):  # numpy arrays have this method
                return obj.tolist()
            
            # Handle Path objects
            elif hasattr(obj, '__fspath__'):  # pathlib.Path objects
                return str(obj)
                
            # Handle other objects by converting to string
            else:
                return str(obj)
                
        except Exception:
            # If all else fails, convert to string
            return str(obj)
    
    def resume_training(self, iteration: int) -> None:
        """Resume training from a specific iteration."""
        self.init_iter = iteration
        self.current_iter = iteration
        self.info(f"Resuming training from iteration {iteration}")
    
    def start_validation(self, total_batches: int):
        """Start validation context with progress tracking."""
        self.is_training = False
        self.info(f"Starting validation with {total_batches} batches")
        
        if self.enable_rich:
            self._start_validation_rich_progress(total_batches)
        else:
            self._start_validation_simple_progress(total_batches)
            
        return self
    
    def _start_validation_rich_progress(self, total_batches: int) -> None:
        """Start rich progress display for validation."""
        self.progress = Progress(
            MofNCompleteColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            "[bold cyan]Images: {task.fields[images]:.0f}",
            "[bold magenta]Instances: {task.fields[instances]:.0f}",
            console=self.console,
            refresh_per_second=4,
        )
        
        self.progress.__enter__()
        self.task_id = self.progress.add_task(
            f"[bold green]Validation {self.experiment_name}",
            total=total_batches,
            completed=0,
            images=0,
            instances=0
        )
    
    def _start_validation_simple_progress(self, total_batches: int) -> None:
        """Start simple tqdm progress for validation."""
        self.progress = tqdm(
            total=total_batches,
            desc=f"Validation {self.experiment_name}",
            unit="batch",
            dynamic_ncols=True,
        )
    
    def validation_batch_step(self, batch_idx: int, images: int, instances: int) -> None:
        """Update validation progress for a batch."""
        if self.progress:
            if self.enable_rich and self.task_id is not None:
                self.progress.update(
                    self.task_id,
                    completed=batch_idx + 1,
                    images=images,
                    instances=instances
                )
            elif self.progress:
                # Update tqdm
                postfix = {"images": images, "instances": instances}
                self.progress.set_postfix(postfix)
                self.progress.update(1)
    
    def __enter__(self):
        """Enter training context."""
        self.is_training = True
        self.info("Starting training session")
        
        if self.enable_rich:
            self._start_rich_progress()
        else:
            self._start_simple_progress()
            
        return self
    
    def _start_rich_progress(self) -> None:
        """Start rich progress display."""
        self.progress = Progress(
            MofNCompleteColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            "[bold blue]Loss: {task.fields[loss]:.4f}",
            "[bold red]LR: {task.fields[lr]:.2e}",
            console=self.console,
            refresh_per_second=4,
        )
        
        self.progress.__enter__()
        self.task_id = self.progress.add_task(
            f"[bold green]{self.experiment_name}",
            total=self.total_iters,
            completed=self.init_iter,
            loss=0.0,
            lr=0.0
        )
    
    def _start_simple_progress(self) -> None:
        """Start simple tqdm progress."""
        self.progress = tqdm(
            total=self.total_iters,
            initial=self.init_iter,
            desc=f"Training {self.experiment_name}",
            unit="iter",
            dynamic_ncols=True,
        )
    
    def step(self, iteration: int, metrics: Dict[str, Union[float, torch.Tensor]], **kwargs) -> None:
        """Log a training step with metrics."""
        self.current_iter = iteration
        
        # Convert tensor metrics to float
        processed_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                processed_metrics[key] = value.item()
            else:
                processed_metrics[key] = float(value)
        
        # Update metrics tracker
        self.metrics_tracker.update(processed_metrics)
        
        # Update progress display
        if self.progress:
            self._update_progress(processed_metrics, **kwargs)
        
        # Log to file periodically
        if iteration % self.save_frequency == 0:
            self._save_metrics_checkpoint(iteration, processed_metrics)
            
        # Log summary periodically
        if iteration % (self.save_frequency * 5) == 0:
            self._log_training_summary(iteration)
    
    def _update_progress(self, metrics: Dict[str, float], **kwargs) -> None:
        """Update progress display."""
        if self.enable_rich and self.task_id is not None:
            # Get current averages
            current_loss = self.metrics_tracker.get_current_avg('loss')
            current_lr = kwargs.get('learning_rate', 0.0)
            
            self.progress.update(
                self.task_id,
                completed=self.current_iter,
                loss=current_loss,
                lr=current_lr
            )
        elif self.progress:
            # Update tqdm
            postfix = {k: f"{v:.4f}" for k, v in metrics.items() if 'loss' in k.lower()}
            if 'learning_rate' in kwargs:
                postfix['lr'] = f"{kwargs['learning_rate']:.2e}"
            self.progress.set_postfix(postfix)
            self.progress.update(1)
    
    def _save_metrics_checkpoint(self, iteration: int, metrics: Dict[str, float]) -> None:
        """Save metrics checkpoint to file."""
        if not is_main_process():
            return
            
        checkpoint_data = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": time.time() - self.start_time,
            "metrics": metrics,
            "averages": {k: self.metrics_tracker.get_current_avg(k) for k in metrics.keys()},
            "experiment": self.experiment_name
        }
        
        try:
            with open(self.metrics_log_file, 'a') as f:
                f.write(json.dumps(checkpoint_data) + '\n')
        except Exception as e:
            self.error(f"Failed to save metrics checkpoint: {e}")
    
    def _log_training_summary(self, iteration: int) -> None:
        """Log comprehensive training summary."""
        elapsed = time.time() - self.start_time
        iter_per_sec = iteration / elapsed if elapsed > 0 else 0
        
        summary_lines = [
            f"Training Summary - Iteration {iteration}/{self.total_iters}",
            f"Elapsed Time: {elapsed/3600:.2f}h | Speed: {iter_per_sec:.2f} iter/s",
        ]
        
        # Add metrics summary
        for metric_name in ['loss', 'mask_loss']:
            if metric_name in self.metrics_tracker.metrics:
                stats = self.metrics_tracker.get_global_stats(metric_name)
                trend = self.metrics_tracker.get_recent_trend(metric_name)
                summary_lines.append(
                    f"{metric_name.title()}: {stats['mean']:.4f}±{stats['std']:.4f} "
                    f"(min: {stats['min']:.4f}, max: {stats['max']:.4f}) [{trend}]"
                )
        
        summary = "\n".join(summary_lines)
        self.info(f"\n{summary}")
    
    def validation_step(self, metrics: Dict[str, Union[float, torch.Tensor]]) -> None:
        """Log validation metrics."""
        processed_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                processed_metrics[key] = value.item()
            else:
                processed_metrics[key] = float(value)
        
        # Log validation metrics
        val_message = "Validation: " + " | ".join([
            f"{k}: {v:.4f}" for k, v in processed_metrics.items()
        ])
        self.info(val_message)
        
        # Save to validation log
        val_data = {
            "iteration": self.current_iter,
            "timestamp": datetime.now().isoformat(),
            "validation_metrics": processed_metrics,
            "experiment": self.experiment_name
        }
        
        if is_main_process():
            val_log_file = self.log_dir / f"{self.experiment_name}_validation.jsonl"
            try:
                with open(val_log_file, 'a') as f:
                    f.write(json.dumps(val_data) + '\n')
            except Exception as e:
                self.error(f"Failed to save validation metrics: {e}")
    
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        # Convert DictConfig to regular dict if needed
        if hasattr(params, '_metadata'):  # DictConfig detection
            try:
                from omegaconf import OmegaConf
                serializable_params = OmegaConf.to_container(params, resolve=True)
            except Exception as e:
                self.warning(f"Failed to convert DictConfig to dict: {e}")
                serializable_params = dict(params)
        else:
            serializable_params = params
            
        self.info("Hyperparameters logged", extra={"hyperparameters": serializable_params})
        
        if is_main_process():
            params_file = self.log_dir / f"{self.experiment_name}_hyperparameters.json"
            try:
                with open(params_file, 'w') as f:
                    json.dump(serializable_params, f, indent=2, default=self._json_serializer)
                self.info(f"Hyperparameters saved to {params_file}")
            except Exception as e:
                self.error(f"Failed to save hyperparameters: {e}")
    
    def log_system_info(self) -> None:
        """Log system information."""
        try:
            import torch
            import platform
            
            system_info = {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
            }
            
            self.info("System Information", extra={"system_info": system_info})
            
            if is_main_process():
                system_file = self.log_dir / f"{self.experiment_name}_system_info.json"
                with open(system_file, 'w') as f:
                    json.dump(system_info, f, indent=2)
                
        except Exception as e:
            self.error(f"Failed to log system info: {e}")
    
    def create_metrics_plot(self, metric_names: List[str] = None) -> None:
        """Create and save metrics plots."""
        try:
            import matplotlib.pyplot as plt
            
            if metric_names is None:
                metric_names = list(self.metrics_tracker.global_metrics.keys())
            
            fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 4 * len(metric_names)))
            if len(metric_names) == 1:
                axes = [axes]
            
            for i, metric_name in enumerate(metric_names):
                values = self.metrics_tracker.global_metrics.get(metric_name, [])
                if values:
                    axes[i].plot(values)
                    axes[i].set_title(f"{metric_name.title()} Over Time")
                    axes[i].set_xlabel("Iteration")
                    axes[i].set_ylabel(metric_name.title())
                    axes[i].grid(True, alpha=0.3)
            
            if is_main_process():
                plt.tight_layout()
                plot_file = self.log_dir / f"{self.experiment_name}_metrics.png"
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                self.info(f"Metrics plot saved to {plot_file}")
            else:
                plt.close()
            
        except Exception as e:
            self.warning(f"Failed to create metrics plot: {e}")
    
    def finish_validation(self):
        """Finish validation context."""
        if self.progress:
            if self.enable_rich:
                self.progress.__exit__(None, None, None)
            else:
                self.progress.close()
            self.progress = None
            self.task_id = None
        
        self.info("Validation completed")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit training context."""
        was_training = self.is_training
        self.is_training = False
        
        if self.progress:
            if self.enable_rich:
                self.progress.__exit__(exc_type, exc_val, exc_tb)
            else:
                self.progress.close()
        
        # Only do training-specific cleanup if we were actually training
        if was_training:
            # Final summary
            total_time = time.time() - self.start_time
            final_iter = self.current_iter
            
            if exc_type is None:
                self.info(f"Training completed successfully!")
                self.info(f"Total iterations: {final_iter}")
                self.info(f"Total time: {total_time/3600:.2f} hours")
                self.info(f"Average speed: {final_iter/total_time:.2f} iterations/second")
            else:
                self.error(f"Training terminated with error: {exc_val}")
            
            # Create final metrics plot
            if final_iter > 0:
                self.create_metrics_plot()
            
            # Save final summary
            self._save_final_summary(final_iter, total_time, exc_type is None)
    
    def _save_final_summary(self, iterations: int, total_time: float, success: bool) -> None:
        """Save final training summary."""
        summary = {
            "experiment": self.experiment_name,
            "success": success,
            "total_iterations": iterations,
            "total_time_seconds": total_time,
            "total_time_hours": total_time / 3600,
            "average_iter_per_second": iterations / total_time if total_time > 0 else 0,
            "final_metrics": {
                name: self.metrics_tracker.get_global_stats(name) 
                for name in self.metrics_tracker.global_metrics.keys()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        if is_main_process():
            summary_file = self.log_dir / f"{self.experiment_name}_summary.json"
            try:
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                self.info(f"Final summary saved to {summary_file}")
            except Exception as e:
                self.error(f"Failed to save final summary: {e}")


# Backwards compatibility alias
Logger = AdvancedLogger
BaseLogger = AdvancedLogger