#!/bin/bash
# Run scene evaluation in parallel for ALL scenes found in the input directory
#
# Usage: ./scripts/run_parallel_all.sh <method> <num_workers> [extra_args...]
#
# Examples:
#   # Evaluate all LayoutVLM scenes with 4 workers
#   ./scripts/run_parallel_all.sh LayoutVLM 4
#
#   # Full example with metrics
#   ./scripts/run_parallel_all.sh SceneWeaver 8 \
#       'evaluation_plan.evaluation_cfg.metrics=[CollisionMetric,StaticEquilibriumMetricCoACD]' \
#       'evaluation_plan.evaluation_cfg.use_empty_matching_result=True'
#
# This script:
#   1. Scans input/<method>/ for all scene_*.json files
#   2. Extracts scene IDs and splits them evenly across workers
#   3. Each worker runs a subset of scenes in parallel
#
# Output:
#   - Each scene creates its own output directory with eval_result.json and eval.log
#   - Worker stdout/stderr is captured in logs/worker_all_*.log

set -e

# Parse arguments
METHOD=${1:-""}
NUM_WORKERS=${2:-4}

if [ $# -lt 2 ] || [ -z "$METHOD" ]; then
    echo "Usage: $0 <method> <num_workers> [extra_args...]"
    echo "Example: $0 LayoutVLM 4"
    echo ""
    echo "Available methods:"
    ls -d input/*/ 2>/dev/null | xargs -I{} basename {} | sed 's/^/  - /'
    exit 1
fi

shift 2  # Remove first two args, rest are passed to main.py

# Create unique run ID and log directory
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="logs/run_${RUN_ID}"
mkdir -p "$LOG_DIR"

# Find all scene IDs from input directory
INPUT_DIR="input/$METHOD"
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Extract scene IDs from scene_*.json files
SCENE_IDS=$(ls "$INPUT_DIR"/scene_*.json 2>/dev/null | \
    sed 's/.*scene_\([0-9]*\)\.json/\1/' | \
    sort -n | \
    tr '\n' ',' | \
    sed 's/,$//')

if [ -z "$SCENE_IDS" ]; then
    echo "Error: No scene_*.json files found in $INPUT_DIR"
    exit 1
fi

# Convert to array
IFS=',' read -ra SCENE_ARRAY <<< "$SCENE_IDS"
TOTAL_SCENES=${#SCENE_ARRAY[@]}

# Calculate scenes per worker
SCENES_PER_WORKER=$(( (TOTAL_SCENES + NUM_WORKERS - 1) / NUM_WORKERS ))  # Ceiling division
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
echo "Parallel Scene Evaluation (All Mode)"
echo "========================================"
echo "Run ID: $RUN_ID"
echo "Method: $METHOD"
echo "Input directory: $INPUT_DIR"
echo "Scene IDs found: $SCENE_IDS"
echo "Total scenes: $TOTAL_SCENES"
echo "Workers: $NUM_WORKERS"
echo "Scenes per worker: ~$SCENES_PER_WORKER"
echo "Extra args: $@"
echo "========================================"
echo ""

# Launch workers
WORKER_COUNT=0
for ((i=0; i<TOTAL_SCENES; i+=SCENES_PER_WORKER)); do
    # Get slice of scenes for this worker
    END=$((i + SCENES_PER_WORKER))
    if [ $END -gt $TOTAL_SCENES ]; then
        END=$TOTAL_SCENES
    fi

    # Build scene list for this worker
    WORKER_SCENES=""
    for ((j=i; j<END; j++)); do
        if [ -n "$WORKER_SCENES" ]; then
            WORKER_SCENES="$WORKER_SCENES,${SCENE_ARRAY[$j]}"
        else
            WORKER_SCENES="${SCENE_ARRAY[$j]}"
        fi
    done

    if [ -z "$WORKER_SCENES" ]; then
        continue
    fi

    echo "Starting worker $WORKER_COUNT: scenes [$WORKER_SCENES]"

    python main.py \
        "evaluation_plan.input_cfg.scene_methods=[$METHOD]" \
        'evaluation_plan.input_cfg.scene_mode=list' \
        "evaluation_plan.input_cfg.scene_list=[$WORKER_SCENES]" \
        "$@" \
        > "${LOG_DIR}/worker_${WORKER_COUNT}.log" 2>&1 &

    PIDS+=($!)
    WORKER_COUNT=$((WORKER_COUNT + 1))
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
