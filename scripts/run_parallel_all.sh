#!/bin/bash
# Run scene evaluation in parallel for ALL scenes found in the input directory
#
# Usage: ./scripts/run_parallel_all.sh <method> <num_workers> [--input-path <path>] [--output-path <path>] [--skip-existing] [--max-retries <n>] [extra_args...]
#
# Options:
#   --skip-existing       Skip scenes that already have complete evaluations
#   --input-path PATH     Custom input directory (default: input/<method>)
#   --output-path PATH    Custom output directory
#   --max-retries N       Max retry attempts per scene on failure (default: 5)
#   --recompute-semantic  Recompute only VLM-based semantic metrics, preserving Drake metrics
#
# Features:
#   - Automatic segfault protection: each scene is processed individually
#   - Failed scenes are automatically retried up to --max-retries times
#   - State files track progress for debugging and manual recovery
#
# Examples:
#   # Evaluate all LayoutVLM scenes with 4 workers
#   ./scripts/run_parallel_all.sh LayoutVLM 4
#
#   # Skip scenes that already have complete evaluations
#   ./scripts/run_parallel_all.sh LayoutVLM 4 --skip-existing
#
#   # With custom max retries (default is 5)
#   ./scripts/run_parallel_all.sh LayoutVLM 4 --max-retries 3
#
#   # With custom output path
#   ./scripts/run_parallel_all.sh LayoutVLM 4 --output-path /path/to/output
#
#   # With custom output path and skip existing
#   ./scripts/run_parallel_all.sh LayoutVLM 4 --output-path /path/to/output --skip-existing
#
#   # With custom input and output paths
#   ./scripts/run_parallel_all.sh LayoutVLM 4 \
#       --input-path /path/to/custom/input/MethodName \
#       --output-path /path/to/output
#
#   # Full example with metrics
#   ./scripts/run_parallel_all.sh SceneWeaver 8 \
#       'evaluation_plan.evaluation_cfg.metrics=[CollisionMetric,StaticEquilibriumMetricCoACD]' \
#       'evaluation_plan.evaluation_cfg.use_empty_matching_result=True'
#
# This script:
#   1. Scans input/<method>/ (or --input-path) for all scene_*.json files
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

# Check for --skip-existing, --output-path, --input-path, --max-retries, and --recompute-semantic flags
SKIP_EXISTING=false
OUTPUT_PATH=""
INPUT_PATH=""
MAX_RETRIES=5
RECOMPUTE_SEMANTIC=false
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --output-path)
            if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
                OUTPUT_PATH="$2"
                shift 2
            else
                echo "Error: --output-path requires a path argument"
                exit 1
            fi
            ;;
        --input-path)
            if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
                INPUT_PATH="$2"
                shift 2
            else
                echo "Error: --input-path requires a path argument"
                exit 1
            fi
            ;;
        --max-retries)
            if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
                MAX_RETRIES="$2"
                shift 2
            else
                echo "Error: --max-retries requires a number"
                exit 1
            fi
            ;;
        --recompute-semantic)
            RECOMPUTE_SEMANTIC=true
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${EXTRA_ARGS[@]}"

# Helper function: Get output_dir - prioritize OUTPUT_PATH, then extra args, then default
get_output_dir() {
    # First check if OUTPUT_PATH is set via --output-path flag
    if [ -n "$OUTPUT_PATH" ]; then
        echo "$OUTPUT_PATH"
        return
    fi
    # Otherwise check extra args for output_dir=...
    local default_dir="./output_eval"
    for arg in "$@"; do
        if [[ "$arg" =~ output_dir=([^[:space:]]+) ]]; then
            echo "${BASH_REMATCH[1]}"
            return
        fi
    done
    echo "$default_dir"
}

# Get the directory where this script is located (for finding helper scripts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Helper function: Check if a scene has a complete evaluation
# Args: $1 = output_dir, $2 = method, $3 = scene_id
is_eval_complete() {
    local output_dir="$1"
    local method="$2"
    local scene_id="$3"
    local eval_file="${output_dir}/${method}/scene_${scene_id}/eval_result.json"

    if [ ! -f "$eval_file" ]; then
        return 1  # No eval file
    fi

    # Use helper script to check if all metrics have results
    # Note: Capture exit code to avoid triggering set -e
    python3 "${SCRIPT_DIR}/check_eval_complete.py" "$eval_file" || return 1
    return 0
}

# Create unique run ID, log directory, and state directory
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="logs/run_${RUN_ID}"
STATE_DIR="${LOG_DIR}/state"
mkdir -p "$LOG_DIR" "$STATE_DIR"

# Find all scene IDs from input directory
if [ -n "$INPUT_PATH" ]; then
    INPUT_DIR="$INPUT_PATH"
else
    INPUT_DIR="input/$METHOD"
fi
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

# Filter out scenes with complete evaluations if --skip-existing is set
if [ "$SKIP_EXISTING" = true ]; then
    OUTPUT_DIR=$(get_output_dir "$@")
    # Use the method name from INPUT_PATH if provided, else use METHOD
    if [ -n "$INPUT_PATH" ]; then
        EVAL_METHOD="$(basename "$INPUT_PATH")"
    else
        EVAL_METHOD="$METHOD"
    fi
    echo "Checking for existing evals in: $OUTPUT_DIR/$EVAL_METHOD"
    FILTERED_SCENES=()
    SKIPPED_COUNT=0
    CHECKED=0
    for scene_id in "${SCENE_ARRAY[@]}"; do
        ((CHECKED++)) || true  # Prevent set -e from triggering on 0
        # Show progress every 20 scenes
        if [ $((CHECKED % 20)) -eq 0 ]; then
            echo "  Checked $CHECKED/$TOTAL_SCENES scenes..."
        fi
        if is_eval_complete "$OUTPUT_DIR" "$EVAL_METHOD" "$scene_id"; then
            ((SKIPPED_COUNT++)) || true  # Prevent set -e from triggering on 0
        else
            FILTERED_SCENES+=("$scene_id")
        fi
    done
    SCENE_ARRAY=("${FILTERED_SCENES[@]}")
    TOTAL_SCENES=${#SCENE_ARRAY[@]}
    echo "Skipping $SKIPPED_COUNT scenes with complete evaluations"
    echo "Remaining scenes to evaluate: $TOTAL_SCENES"

    if [ $TOTAL_SCENES -eq 0 ]; then
        echo "All scenes already have complete evaluations. Nothing to do."
        trap - EXIT
        exit 0
    fi

    # Update SCENE_IDS for display
    SCENE_IDS=$(IFS=','; echo "${SCENE_ARRAY[*]}")
fi

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
echo "Output directory: $(get_output_dir "$@")"
echo "Scene IDs found: $SCENE_IDS"
echo "Total scenes: $TOTAL_SCENES"
echo "Workers: $NUM_WORKERS"
echo "Scenes per worker: ~$SCENES_PER_WORKER"
echo "Max retries per scene: $MAX_RETRIES"
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

    # Build input path args if custom path provided
    INPUT_ARGS=""
    if [ -n "$INPUT_PATH" ]; then
        INPUT_ROOT_DIR="$(dirname "$INPUT_PATH")"
        INPUT_METHOD="$(basename "$INPUT_PATH")"
        INPUT_ARGS="evaluation_plan.input_cfg.root_dir=$INPUT_ROOT_DIR evaluation_plan.input_cfg.scene_methods=[$INPUT_METHOD]"
    else
        INPUT_ARGS="evaluation_plan.input_cfg.scene_methods=[$METHOD]"
    fi

    # Build output path argument if set
    OUTPUT_PATH_ARG=""
    if [ -n "$OUTPUT_PATH" ]; then
        OUTPUT_PATH_ARG="evaluation_plan.evaluation_cfg.output_dir=$OUTPUT_PATH"
    fi

    # Build recompute-semantic argument if set
    RECOMPUTE_SEMANTIC_ARG=""
    if [ "$RECOMPUTE_SEMANTIC" = true ]; then
        RECOMPUTE_SEMANTIC_ARG="evaluation_plan=recompute_vlm_plan"
    fi

    # Use fault-tolerant worker wrapper that processes scenes one-by-one with retry logic
    "${SCRIPT_DIR}/run_worker_with_restart.sh" \
        "$WORKER_COUNT" \
        "$WORKER_SCENES" \
        "$STATE_DIR" \
        "$MAX_RETRIES" \
        $RECOMPUTE_SEMANTIC_ARG \
        $INPUT_ARGS \
        $OUTPUT_PATH_ARG \
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

# Summarize results from state files
echo ""
echo "Summary by worker:"
TOTAL_COMPLETED=0
TOTAL_FAILED=0
for state_file in "$STATE_DIR"/worker_*_state.json; do
    if [ -f "$state_file" ]; then
        WORKER=$(basename "$state_file" | sed 's/worker_\([0-9]*\)_state.json/\1/')
        COMPLETED=$(jq '.completed | length' "$state_file")
        FAILED_SCENES=$(jq '.failed | length' "$state_file")
        TOTAL_COMPLETED=$((TOTAL_COMPLETED + COMPLETED))
        TOTAL_FAILED=$((TOTAL_FAILED + FAILED_SCENES))
        echo "  Worker $WORKER: $COMPLETED completed, $FAILED_SCENES failed"
    fi
done
echo ""
echo "Total: $TOTAL_COMPLETED completed, $TOTAL_FAILED failed"
echo ""

if [ $FAILED -eq 1 ]; then
    echo "SOME WORKERS HAD FAILURES"
    echo ""
    # Show failed scenes from all workers
    echo "Failed scenes:"
    for state_file in "$STATE_DIR"/worker_*_state.json; do
        if [ -f "$state_file" ]; then
            jq -r '.failed[] | "  Scene \(.scene) (after \(.attempts) attempts)"' "$state_file" 2>/dev/null
        fi
    done
    echo ""
    echo "Check ${LOG_DIR}/worker_*.log for details"
    echo "State files: ${STATE_DIR}/"
    exit 1
else
    echo "ALL SCENES COMPLETED SUCCESSFULLY"
fi
echo "Logs: ${LOG_DIR}/"
echo "State files: ${STATE_DIR}/"
echo "========================================"
