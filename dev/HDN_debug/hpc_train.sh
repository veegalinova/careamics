#!/bin/bash
[ -z "$1" ] && { echo "Usage: $0 <script_path>"; exit 1; }

SCRIPT_PATH="$(realpath "$1")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
SCRIPT_NAME="$(basename "$SCRIPT_PATH")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"
SCRIPT_REL="${SCRIPT_PATH#$PROJECT_ROOT/}"
LOGS_DIR="$(cd "$SCRIPT_DIR" && pwd)/logs"
mkdir -p "$LOGS_DIR"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${SCRIPT_NAME%.*}
#SBATCH --partition=gpuq
#SBATCH --output=${LOGS_DIR}/%j.out
#SBATCH --error=${LOGS_DIR}/%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00

export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID
SCRATCH_DIR="\$TMPDIR/careamics_\${SLURM_JOB_ID}"
mkdir -p "\$SCRATCH_DIR"

cd ${PROJECT_ROOT} && tar -cf - . | (cd "\$SCRATCH_DIR" && tar -xf -)
cd "\$SCRATCH_DIR"

[ ! -f pyproject.toml ] && { echo "Error: pyproject.toml not found"; exit 1; }
command -v uv &> /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; source \$HOME/.cargo/env 2>/dev/null; }
UV_CACHE_DIR="\$TMPDIR/.cache/" uv sync

cd "$(dirname "$SCRIPT_REL")" && uv run python "$SCRIPT_NAME"
EOF
