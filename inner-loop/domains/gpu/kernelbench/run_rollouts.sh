#!/bin/bash
# Run autoresearch rollouts on a batch of L1 tasks.
# Usage: bash run_rollouts.sh [num_iters] [model]

set -e

ITERS=${1:-10}
MODEL=${2:-claude-sonnet-4-20250514}
TASK_DIR="/root/KernelBench/KernelBench/level1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Easy starter tasks: elementwise ops where Triton should clearly win
TASKS=(
    "19_ReLU.py"
    "21_Sigmoid.py"
    "22_Tanh.py"
    "26_GELU_.py"
    "29_Softplus.py"
    "30_Softsign.py"
    "47_Sum_reduction_over_a_dimension.py"
    "1_Square_matrix_multiplication_.py"
    "7_Matmul_with_small_K_dimension_.py"
    "39_L2Norm_.py"
)

echo "=== KernelBench Autoresearch Rollouts ==="
echo "Tasks: ${#TASKS[@]}"
echo "Iters per task: $ITERS"
echo "Model: $MODEL"
echo ""

cd "$SCRIPT_DIR"

for task in "${TASKS[@]}"; do
    task_path="$TASK_DIR/$task"
    if [ ! -f "$task_path" ]; then
        echo "SKIP: $task (not found)"
        continue
    fi
    echo ">>> Starting: $task"
    uv run python -u autoresearch_loop.py "$task_path" --iters "$ITERS" --model "$MODEL" 2>&1 | tee -a "rollout_$(date +%Y%m%d).log"
    echo ""
done

echo "=== All rollouts complete ==="
