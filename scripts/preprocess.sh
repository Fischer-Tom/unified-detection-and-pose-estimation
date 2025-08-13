#!/usr/bin/env bash
set -euo pipefail

# ---------- Config ----------
CODE_DIR="${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)}"
DATASET="${DATASET:-REAL275}"
DATA_DIR="${DATA_DIR:-./data/$DATASET}"

# ---------- Helpers ----------

validate_config() {
  echo "=== Configuration Validation ==="
  echo "Code directory: $CODE_DIR"
  echo "Data directory: $DATA_DIR" 
  echo "Dataset: $DATASET"
  echo
  
  if [[ ! -d "$CODE_DIR" ]]; then
    echo "L Error: Code directory not found: $CODE_DIR"
    exit 1
  fi
  
  if [[ ! -d "$DATA_DIR" ]]; then
    echo "L Error: Data directory not found: $DATA_DIR"
    exit 1
  fi
  
  if ! command -v docker &>/dev/null; then
    echo "L Error: Docker not found. Please install Docker with GPU support."
    exit 1
  fi
  
  echo " Configuration validated successfully"
  echo
}

show_help() {
  cat <<EOF
Usage: $0 [OPTIONS]

Preprocess data for unified detection and pose estimation model.

OPTIONS:
  -h, --help          Show this help message

ENVIRONMENT VARIABLES:
  DATASET             Dataset name (default: REAL275)
  DATA_DIR            Path to dataset directory (default: ./data/\$DATASET)
  CODE_DIR            Path to code directory (default: script directory)

EXAMPLES:
  ./preprocess.sh

EOF
}

run_preprocessing() {
  echo "Preprocessing data..."
  docker run --rm \
    -v "$(pwd)":/workspace \
    -v "$DATA_DIR":/hnvme \
    --network host --ipc host \
    --gpus all \
    --workdir /workspace \
    -e PYTHONPATH="${PYTHONPATH:-}:/workspace" \
    fischeto/upose \
    python3 src/uni_dp/preprocess_annos.py --data_path /hnvme
}

main() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help) show_help; exit 0 ;;
      *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
  done

  echo "=== Preprocessing Script ==="
  echo "Dataset: $DATASET"
  echo "Data:    $DATA_DIR"
  echo

  validate_config
  cd "$CODE_DIR"
  
  run_preprocessing
  echo "Preprocessing completed at $(date +%F" "%T.%3N)"
}

main "$@"