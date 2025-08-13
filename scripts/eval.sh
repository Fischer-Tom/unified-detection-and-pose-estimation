#!/usr/bin/env bash
set -euo pipefail

# ---------- Config ----------
# Use script directory as default, or override with environment variable
CODE_DIR="${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)}"
DATASET="${DATASET:-REAL275}"
# Use relative data directory by default, or override with environment variable
DATA_DIR="${DATA_DIR:-./data/$DATASET}"
# Default to weights directory or use environment override
RUNDIR="${RUNDIR:-$CODE_DIR/weights}"
# Default checkpoint name
CHKPT="${CHKPT:-Real}"

validate_config() {
  echo "=== Configuration Validation ==="
  echo "Code directory: $CODE_DIR"
  echo "Data directory: $DATA_DIR"
  echo "Model directory: $RUNDIR"
  echo "Checkpoint: $CHKPT.pt"
  echo "Dataset: $DATASET"
  echo
  
  # Validate code directory
  if [[ ! -d "$CODE_DIR" ]]; then
    echo "❌ Error: Code directory not found: $CODE_DIR"
    echo "💡 Tip: Set CODE_DIR environment variable or run from project root"
    exit 1
  fi
  
  # Validate data directory
  if [[ ! -d "$DATA_DIR" ]]; then
    echo "❌ Error: Data directory not found: $DATA_DIR"
    echo "💡 Expected structure: $DATA_DIR/{real_data,val_models,train_*} for REAL275"
    echo "💡 Set DATA_DIR environment variable or create: mkdir -p $DATA_DIR"
    exit 1
  fi
  
  # Validate model directory
  if [[ ! -d "$RUNDIR" ]]; then
    echo "❌ Error: Model directory not found: $RUNDIR"
    echo "💡 Tip: Set RUNDIR environment variable or use --rundir option"
    echo "💡 Example: ./eval.sh --rundir ./outputs/2024-01-01/12-00-00"
    exit 1
  fi
  
  # Validate checkpoint file
  if [[ ! -f "$RUNDIR/$CHKPT.pt" ]]; then
    echo "❌ Error: Checkpoint not found: $RUNDIR/$CHKPT.pt"
    echo "💡 Available checkpoints in $RUNDIR:"
    if ls "$RUNDIR"/*.pt 1>/dev/null 2>&1; then
      ls -1 "$RUNDIR"/*.pt | sed 's|.*/||; s|\.pt$||' | sed 's|^|   - |'
    else
      echo "   No .pt files found"
    fi
    echo "💡 Use --model option: ./eval.sh --model ModelName"
    exit 1
  fi
  
  # Check Docker availability
  if ! command -v docker &>/dev/null; then
    echo "❌ Error: Docker not found. Please install Docker with GPU support."
    exit 1
  fi
  
  echo "✅ Configuration validated successfully"
  echo
}

show_help() {
  cat <<EOF
Usage: $0 [OPTIONS]

Evaluate unified detection and pose estimation model using Docker.

OPTIONS:
  -m, --model MODEL     Model checkpoint name without .pt extension (default: Real)
  -r, --rundir DIR      Path to model directory (default: ./weights)
  -h, --help            Show this help message

ENVIRONMENT VARIABLES:
  DATASET               Dataset name (default: REAL275, options: REAL275, H6D)
  DATA_DIR              Path to dataset directory (default: ./data/\$DATASET)
  CODE_DIR              Path to code directory (default: script directory)
  RUNDIR                Path to model directory (default: ./weights)
  CHKPT                 Model checkpoint name (default: Real)

EXAMPLES:
  # Basic evaluation with default model
  ./eval.sh

  # Evaluate specific model
  ./eval.sh --model UniDP_149999

  # Evaluate with custom paths
  DATA_DIR=/path/to/data ./eval.sh --rundir ./outputs/2024-01-01/12-00-00 --model MyModel

  # Evaluate H6D dataset
  DATASET=H6D DATA_DIR=/path/to/h6d ./eval.sh

  # Evaluate from training output directory
  ./eval.sh --rundir ./outputs/2024-01-01/12-00-00 --model UniDP_149999

REQUIREMENTS:
  - Docker with GPU support
  - Trained model checkpoint in RUNDIR
  - Dataset in DATA_DIR following expected structure
  - Built Docker image 'fischeto/upose' or build with: docker build -t fischeto/upose .

OUTPUT:
  - Evaluation results saved in model directory
  - ADD(-S) metrics and per-object statistics
  - Result visualization (if enabled)

EOF
}

run_docker() {
  local command="$1"
  docker run --rm -it \
    -v "$(pwd)":/workspace \
    -v "$DATA_DIR":/hnvme \
    -v "$RUNDIR":/run_dir \
    --network host --ipc host \
    --workdir /workspace \
    -e PYTHONPATH="${PYTHONPATH:-}:/workspace" \
    --gpus all \
    fischeto/upose \
    python3 src/uni_dp/eval.py \
    --load_path /run_dir \
    --model_path "$command"
}

run_evaluation() {
  local model_path="$CHKPT.pt"
  
  echo "=== Evaluation Started ==="
  echo "Model: $RUNDIR/$model_path"
  echo "Dataset: $DATASET"
  echo "Data: $DATA_DIR"
  echo "Started at: $(date +%F" "%T.%3N)"
  echo
  
  if run_docker "$model_path"; then
    echo "✅ Evaluation completed successfully"
  else
    echo "❌ Evaluation failed or was interrupted"
    return 1
  fi
}

main() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -m|--model) CHKPT="$2"; shift 2 ;;
      -r|--rundir) RUNDIR="$2"; shift 2 ;;
      -h|--help) show_help; exit 0 ;;
      *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
  done

  validate_config
  
  cd "$CODE_DIR"
  echo "Working from: $(pwd)"
  echo

  run_evaluation

  echo
  echo "Finished at $(date +%F" "%T.%3N)"
}

main "$@"
