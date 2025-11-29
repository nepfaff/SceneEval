#!/bin/bash
# Run scene evaluation in parallel for a specific list of scene IDs
#
# Usage: ./scripts/run_parallel_list.sh <scene_ids> <num_workers> [extra_args...]
#
# Examples:
#   # Evaluate specific scenes with 3 workers
#   ./scripts/run_parallel_list.sh "39,56,74,94,106" 3 'evaluation_plan.input_cfg.scene_methods=[LayoutVLM]'
#
#   # Full example with metrics
#   ./scripts/run_parallel_list.sh "0,5,10,15,20" 5 \
#       'evaluation_plan.input_cfg.scene_methods=[SceneWeaver]' \
#       'evaluation_plan.evaluation_cfg.metrics=[CollisionMetric,StaticEquilibriumMetricCoACD]'
#
# Output:
#   - Each scene creates its own output directory with eval_result.json and eval.log
#   - Worker stdout/stderr is captured in logs/worker_list_*.log

set -e

# Parse arguments
SCENE_IDS=${1:-""}
NUM_WORKERS=${2:-4}

if [ $# -lt 2 ] || [ -z "$SCENE_IDS" ]; then
    echo "Usage: $0 <scene_ids> <num_workers> [extra_args...]"
    echo "Example: $0 \"39,56,74,94,106\" 3 'evaluation_plan.input_cfg.scene_methods=[LayoutVLM]'"
    exit 1
fi

shift 2  # Remove first two args, rest are passed to main.py

# Create logs directory
mkdir -p logs

# Convert comma-separated string to array
IFS=',' read -ra SCENE_ARRAY <<< "$SCENE_IDS"
TOTAL_SCENES=${#SCENE_ARRAY[@]}

# Calculate scenes per worker
SCENES_PER_WORKER=$(( (TOTAL_SCENES + NUM_WORKERS - 1) / NUM_WORKERS ))  # Ceiling division
PIDS=()

echo "========================================"
echo "Parallel Scene Evaluation (List Mode)"
echo "========================================"
echo "Scene IDs: $SCENE_IDS"
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
        'evaluation_plan.input_cfg.scene_mode=list' \
        "evaluation_plan.input_cfg.scene_list=[$WORKER_SCENES]" \
        "$@" \
        > "logs/worker_list_${WORKER_COUNT}.log" 2>&1 &

    PIDS+=($!)
    WORKER_COUNT=$((WORKER_COUNT + 1))
done

echo ""
echo "All workers launched. PIDs: ${PIDS[*]}"
echo "Logs: logs/worker_list_*.log"
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
if [ $FAILED -eq 1 ]; then
    echo "SOME WORKERS FAILED: ${FAILED_WORKERS[*]}"
    echo "Check logs/worker_list_*.log for details"
    exit 1
else
    echo "ALL WORKERS COMPLETED SUCCESSFULLY"
fi
echo "========================================"
