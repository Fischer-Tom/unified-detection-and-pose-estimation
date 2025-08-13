#!/usr/bin/env bash
set -euo pipefail

# ---------- Config ----------
# Use script directory as default, or override with environment variable
CODE_DIR="${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)}"
DATASET="${DATASET:-REAL275}"
# Use relative data directory by default, or override with environment variable  
DATA_DIR="${DATA_DIR:-./data/$DATASET}"
RESUME_TRAINING=""
PORT=29500
N_GPU=""                      # auto-detected

# ---------- State ----------
TRAINING_STARTED=false
CHILD_PID=""
CHILD_PGID=""
TERM_SENT=false

# ---------- Signals ----------
setup_signal_handlers() {
  # USR1: early warning -> send to PID only (avoid killing DataLoader workers)
  forward_usr1_pid() {
   echo "[wrapper] Caught SIGUSR1 at $(date +%F" "%T.%3N)"
    if [[ -n "$CHILD_PID" ]]; then
      echo "[wrapper] Forwarding SIGUSR1 to PID $CHILD_PID"
      kill -s SIGUSR1 "$CHILD_PID" || true
    elif [[ -n "$CHILD_PGID" ]]; then
      echo "[wrapper] (fallback) Forwarding SIGUSR1 to PGID $CHILD_PGID"
      kill -s SIGUSR1 -"$CHILD_PGID" || true
    fi
  }
  trap 'forward_usr1_pid SIGUSR1' SIGUSR1

  # TERM/INT/QUIT: final stop -> broadcast to process group
  forward_term_group() {
    local sig="$1"
    echo "[wrapper] Caught $sig at $(date +%F" "%T.%3N)"
    if [[ "$TERM_SENT" == true ]]; then return; fi
    TERM_SENT=true
    if [[ -n "$CHILD_PGID" ]]; then
      echo "[wrapper] Forwarding $sig to PGID $CHILD_PGID"
      kill -s "$sig" -"$CHILD_PGID" || true
    elif [[ -n "$CHILD_PID" ]]; then
      echo "[wrapper] (fallback) Forwarding $sig to PID $CHILD_PID"
      kill -s "$sig" "$CHILD_PID" || true
    fi
  }
  trap 'forward_term_group SIGTERM' SIGTERM
  trap 'forward_term_group SIGINT'  SIGINT
  trap 'forward_term_group SIGQUIT' SIGQUIT
}

# ---------- Helpers ----------

detect_gpus() {
  if command -v nvidia-smi &>/dev/null; then
    N_GPU="$(nvidia-smi --list-gpus | wc -l || echo 0)"
    if [[ "${N_GPU:-0}" -eq 0 ]]; then echo "Warning: No NVIDIA GPUs detected, using CPU"; N_GPU=1
    else echo "Detected $N_GPU NVIDIA GPU(s)"
    fi
  else
    echo "Warning: nvidia-smi not found, assuming CPU"; N_GPU=1
  fi
  if [[ -n "${FORCE_N_GPU:-}" ]]; then
    echo "Overriding GPU count from $N_GPU to $FORCE_N_GPU"; N_GPU="$FORCE_N_GPU"
  fi
}

validate_config() {
  echo "=== Configuration Validation ==="
  echo "Code directory: $CODE_DIR"
  echo "Data directory: $DATA_DIR" 
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
  
  # Check Docker availability
  if ! command -v docker &>/dev/null; then
    echo "❌ Error: Docker not found. Please install Docker with GPU support."
    exit 1
  fi
  
  detect_gpus
  echo "✅ Configuration validated successfully"
  echo
}

show_help() {
  cat <<EOF
Usage: $0 [OPTIONS]

Train unified detection and pose estimation model using Docker.

OPTIONS:
  -h, --help          Show this help message

ENVIRONMENT VARIABLES:
  DATASET             Dataset name (default: REAL275, options: REAL275, H6D (EXPERIMENTAL!)
  DATA_DIR            Path to dataset directory (default: ./data/\$DATASET)
  CODE_DIR            Path to code directory (default: script directory)
  RESUME_TRAINING     Path to checkpoint directory for resuming
  FORCE_N_GPU         Override GPU count (default: auto-detect)
  PORT                Master port for multi-GPU training (default: 29500)

EXAMPLES:
  # Basic training with defaults
  ./train.sh

  # Resume training from checkpoint
  RESUME_TRAINING=./outputs/2024-01-01/12-00-00 ./train.sh

  # Force single GPU training
  FORCE_N_GPU=1 ./train.sh

REQUIREMENTS:
  - Docker with GPU support
  - Dataset in DATA_DIR following expected structure
  - Built Docker image 'fischeto/upose' or build with: docker build -t fischeto/upose .

EOF
}

# Launch docker in its own session; capture real PGID via ps
run_docker() {
  local command="$1"
  setsid docker run --rm \
    -v "$(pwd)":/workspace \
    -v "$DATA_DIR":/hnvme \
    --network host --ipc host \
    --gpus all \
    --workdir /workspace \
    -e PYTHONPATH="${PYTHONPATH:-}:/workspace" \
    fischeto/upose \
    $command &
  CHILD_PID=$!
  # real PGID (don't assume PGID==PID)
  CHILD_PGID="$(ps -o pgid= "$CHILD_PID" | tr -d '[:space:]')"
  echo "[wrapper] docker PID=$CHILD_PID PGID=$CHILD_PGID"

  # Optional watchdog after TERM (adjust 550s to your USR1 lead time if desired)
  (
    sleep 550
    if [[ "$TERM_SENT" == true && -n "$CHILD_PGID" ]]; then
      echo "[wrapper] Watchdog: SIGKILL PGID $CHILD_PGID"
      kill -s SIGKILL -"$CHILD_PGID" 2>/dev/null || true
    fi
  ) &
  local rc
  while true; do
    wait "$CHILD_PID"; rc=$?
    if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
      continue   # was interrupted by a signal; keep waiting
    fi
    break
  done
  CHILD_PID=""; CHILD_PGID=""
  return $rc
}


run_training() {
  local train_cmd="src/uni_dp/train.py params=$DATASET dataset=$DATASET"
  if [[ "${N_GPU:-1}" -gt 1 ]]; then
    train_cmd="$train_cmd ddp.distributed=true ddp.world_size=$N_GPU"
    train_cmd="torchrun --master_port $PORT --nproc_per_node=$N_GPU --max-restarts=0 $train_cmd"
    echo "Starting multi-GPU training with $N_GPU GPUs"
  else
    train_cmd="python3 $train_cmd ddp.distributed=false ddp.world_size=1"
    echo "Starting single-GPU/CPU training"
  fi
  if [[ -n "$RESUME_TRAINING" ]]; then
    train_cmd="$train_cmd hydra.run.dir=$RESUME_TRAINING"
    echo "Resuming from $RESUME_TRAINING at $(date +%F" "%T.%3N)"
  else
    echo "Start training at $(date +%F" "%T.%3N)"
  fi
  echo "Command: $train_cmd"

  TRAINING_STARTED=true
  if run_docker "$train_cmd"; then
    TRAINING_STARTED=false; return 0
  else
    TRAINING_STARTED=false; return 1
  fi
}

main() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help) show_help; exit 0 ;;
      *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
  done

  echo "=== Training Script ==="
  echo "Dataset: $DATASET"
  echo "Code:    $CODE_DIR"
  echo "Data:    $DATA_DIR"
  echo

  setup_signal_handlers
  validate_config

  cd "$CODE_DIR"
  echo "Working from: $(pwd)"

  if run_training; then
    echo "Training finished normally."
  else
    echo "Training failed or was interrupted."
  fi

  echo "Finished at $(date +%F" "%T.%3N)"
}

main "$@"