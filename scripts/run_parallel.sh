#!/bin/bash
# Run scene evaluation in parallel by splitting scene ranges across processes
#
# Usage: ./scripts/run_parallel.sh <total_scenes> <num_workers> [extra_args...]
#
# Examples:
#   # Evaluate 100 scenes with 4 workers
#   ./scripts/run_parallel.sh 100 4 'evaluation_plan.input_cfg.scene_methods=[SceneWeaver]'
#
#   # Full example with metrics
#   ./scripts/run_parallel.sh 100 4 \
#       'evaluation_plan.input_cfg.scene_methods=[SceneWeaver]' \
#       'evaluation_plan.evaluation_cfg.metrics=[CollisionMetric,StaticEquilibriumMetricCoACD]' \
#       'evaluation_plan.evaluation_cfg.use_empty_matching_result=True'
#
# Output:
#   - Each scene creates its own output directory with eval_result.json and eval.log
#   - Worker stdout/stderr is captured in logs/worker_*.log

set -e

# Parse arguments
TOTAL_SCENES=${1:-100}
NUM_WORKERS=${2:-4}

if [ $# -lt 2 ]; then
    echo "Usage: $0 <total_scenes> <num_workers> [extra_args...]"
    echo "Example: $0 100 4 'evaluation_plan.input_cfg.scene_methods=[SceneWeaver]'"
    exit 1
fi

shift 2  # Remove first two args, rest are passed to main.py

# Create unique run ID and log directory
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="logs/run_${RUN_ID}"
mkdir -p "$LOG_DIR"

# Calculate scenes per worker
SCENES_PER_WORKER=$((TOTAL_SCENES / NUM_WORKERS))
PIDS=()

# Cleanup function to kill all workers
cleanup() {
    echo ""
    echo "Caught signal, terminating workers..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
        fi
    done
    # Wait briefly for graceful shutdown, then force kill
    sleep 1
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null
        fi
    done
    echo "Workers terminated."
    exit 130
}

# Trap signals to ensure cleanup
trap cleanup INT TERM EXIT

echo "========================================"
echo "Parallel Scene Evaluation"
echo "========================================"
echo "Run ID: $RUN_ID"
echo "Total scenes: $TOTAL_SCENES"
echo "Workers: $NUM_WORKERS"
echo "Scenes per worker: ~$SCENES_PER_WORKER"
echo "Extra args: $@"
echo "========================================"
echo ""

# Launch workers
for ((i=0; i<NUM_WORKERS; i++)); do
    START=$((i * SCENES_PER_WORKER))
    if [ $i -eq $((NUM_WORKERS - 1)) ]; then
        END=$TOTAL_SCENES  # Last worker gets remainder
    else
        END=$(((i + 1) * SCENES_PER_WORKER))
    fi

    echo "Starting worker $i: scenes [$START, $END)"

    python main.py \
        'evaluation_plan.input_cfg.scene_mode=range' \
        "evaluation_plan.input_cfg.scene_range=[$START,$END]" \
        "$@" \
        > "${LOG_DIR}/worker_${i}.log" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All workers launched. PIDs: ${PIDS[*]}"
echo "Logs: ${LOG_DIR}/worker_*.log"
echo ""
echo "Waiting for completion..."

# Wait for all workers and track failures
FAILED=0
FAILED_WORKERS=()

for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if ! wait $PID; then
        echo "Worker $i (PID $PID) FAILED"
        FAILED=1
        FAILED_WORKERS+=($i)
    else
        echo "Worker $i (PID $PID) completed"
    fi
done

echo ""
echo "========================================"

# Disable EXIT trap for normal completion
trap - EXIT

if [ $FAILED -eq 1 ]; then
    echo "SOME WORKERS FAILED: ${FAILED_WORKERS[*]}"
    echo "Check ${LOG_DIR}/worker_*.log for details"
    exit 1
else
    echo "ALL WORKERS COMPLETED SUCCESSFULLY"
fi
echo "Logs: ${LOG_DIR}/"
echo "========================================"
