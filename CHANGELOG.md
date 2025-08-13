# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-08-09

### Added
- Initial release of Unified Detection and Pose Estimation framework
- Neural mesh memory for 6D pose estimation
- Support for REAL275 and H6D datasets
- Multi-GPU training with automatic GPU detection
- Docker containerization for reproducible training
- SLURM cluster support with robust signal handling
- Comprehensive logging and metrics tracking
- Structured configuration system using Hydra
- Evaluation pipeline with ADD(-S) metrics

### Features
- **Training**: Distributed training with DDP support
- **Datasets**: REAL275 (6 objects), H6D (experimental)
- **Models**: UniDP architecture with neural mesh representations
- **Containers**: Docker for local development
- **Monitoring**: Rich logging, metrics visualization, WandB integration
- **Scripts**: Automated training and evaluation workflows

### Technical Details
- PyTorch 2.0+ with CUDA 12.6 support
- Automatic mixed precision training
- Gradient clipping and learning rate scheduling
- Checkpoint saving and resumption
- Signal handling for graceful termination

## [Unreleased]

### Planned
- Extended dataset support
- Model architecture improvements
- Performance optimizations
- Additional evaluation metrics