#!/bin/bash
# Batch recompute VLM metrics for multiple methods
#
# This script orchestrates VLM metric recomputation across multiple methods
# using a worker pool. It preserves existing Drake metrics (expensive physics
# simulations) while recomputing only VLM-based semantic metrics.
#
# Usage: ./scripts/run_recompute_vlm_all_methods.sh <num_workers> [options] <method_mappings...>
#
# Arguments:
#   num_workers      Number of parallel workers (e.g., 192)
#   method_mappings  One or more "input_name" or "input_name:output_name" pairs
#                    If output_name is omitted, uses input_name
#
# Options:
#   --skip-existing  Skip scenes where eval_result_v2.json already exists
#   --dry-run        Show what would be processed without running
#   --max-retries N  Max retry attempts per scene on failure (default: 3)
#
# Fixed directories:
#   Input:  /home/ubuntu/efs/nicholas/scene-agent-eval-scenes/SceneEval_converted/
#   Output: /home/ubuntu/efs/nicholas/scene-agent-eval-scenes/SceneEval_results/
#
# Prerequisites:
#   - Each scene must have an existing eval_result.json (contains Drake metrics to preserve)
#   - Scenes without eval_result.json are skipped with a warning
#
# Examples:
#   # Run all SceneAgent variants with 192 workers
#   ./scripts/run_recompute_vlm_all_methods.sh 192 \
#       SceneAgent_Ours_Room \
#       SceneAgent_Ours_House \
#       SceneAgent_NoCritic
#
#   # With different input/output names
#   ./scripts/run_recompute_vlm_all_methods.sh 192 \
#       "LayoutVLM_Curated_Fixed:LayoutVLM_Curated_Fixed"
#
#   # Skip already completed scenes
#   ./scripts/run_recompute_vlm_all_methods.sh 192 --skip-existing \
#       SceneAgent_Ours_Room
#
#   # Dry run to see what would be processed
#   ./scripts/run_recompute_vlm_all_methods.sh 192 --dry-run --skip-existing \
#       SceneAgent_Ours_Room SceneAgent_Ours_House

set -e

# Fixed parent directories
INPUT_PARENT="/home/ubuntu/efs/nicholas/scene-agent-eval-scenes/SceneEval_converted"
OUTPUT_PARENT="/home/ubuntu/efs/nicholas/scene-agent-eval-scenes/SceneEval_results"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Helper function to find the actual scene directory
# Some methods have nested structure: OUTPUT_PARENT/OUTPUT_NAME/INPUT_NAME/scene_*/
# Others have flat structure: OUTPUT_PARENT/OUTPUT_NAME/scene_*/
# Returns the base path (without scene_N) or empty string if not found
# Always returns 0 to avoid triggering set -e
find_scene_base_dir() {
    local output_dir="$1"
    local input_name="$2"
    local scene_id="$3"

    # Try flat structure first (most common)
    if [ -d "${output_dir}/scene_${scene_id}" ]; then
        echo "${output_dir}"
        return 0
    fi

    # Try nested structure (LayoutVLM, etc.)
    if [ -d "${output_dir}/${input_name}/scene_${scene_id}" ]; then
        echo "${output_dir}/${input_name}"
        return 0
    fi

    # Not found - return empty string (caller checks for empty)
    echo ""
    return 0
}

# Parse arguments
NUM_WORKERS=${1:-""}

if [ -z "$NUM_WORKERS" ] || ! [[ "$NUM_WORKERS" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 <num_workers> [options] <method_mappings...>"
    echo "Example: $0 192 --skip-existing SceneAgent_Ours_Room SceneAgent_Ours_House"
    exit 1
fi

shift 1

# Parse options and method mappings
SKIP_EXISTING=false
DRY_RUN=false
MAX_RETRIES=3
METHOD_MAPPINGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
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
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            METHOD_MAPPINGS+=("$1")
            shift
            ;;
    esac
done

if [ ${#METHOD_MAPPINGS[@]} -eq 0 ]; then
    echo "Error: No method mappings provided"
    echo ""
    echo "Available methods in $INPUT_PARENT:"
    ls -d "$INPUT_PARENT"/*/ 2>/dev/null | xargs -I{} basename {} | sed 's/^/  - /'
    exit 1
fi

# Create unique run ID, log directory, and state directory
RUN_ID="vlm_recompute_$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="logs/run_${RUN_ID}"
STATE_DIR="${LOG_DIR}/state"
mkdir -p "$LOG_DIR" "$STATE_DIR"

echo "========================================"
echo "VLM Metric Recomputation (All Methods)"
echo "========================================"
echo "Run ID: $RUN_ID"
echo "Workers: $NUM_WORKERS"
echo "Max retries: $MAX_RETRIES"
echo "Skip existing: $SKIP_EXISTING"
echo "Dry run: $DRY_RUN"
echo ""
echo "Input parent: $INPUT_PARENT"
echo "Output parent: $OUTPUT_PARENT"
echo ""
echo "Methods to process:"
for mapping in "${METHOD_MAPPINGS[@]}"; do
    IFS=':' read -r INPUT_NAME OUTPUT_NAME <<< "$mapping"
    OUTPUT_NAME="${OUTPUT_NAME:-$INPUT_NAME}"
    echo "  - $INPUT_NAME -> $OUTPUT_NAME"
done
echo "========================================"
echo ""

# Collect all scenes across methods
ALL_SCENES=()
SKIPPED_V2_EXISTS=()
SKIPPED_NO_EVAL=()

echo "Scanning scenes..."

for mapping in "${METHOD_MAPPINGS[@]}"; do
    # Parse mapping: input_name:output_name (output defaults to input)
    IFS=':' read -r INPUT_NAME OUTPUT_NAME <<< "$mapping"
    OUTPUT_NAME="${OUTPUT_NAME:-$INPUT_NAME}"

    INPUT_DIR="${INPUT_PARENT}/${INPUT_NAME}"
    OUTPUT_DIR="${OUTPUT_PARENT}/${OUTPUT_NAME}"

    if [ ! -d "$INPUT_DIR" ]; then
        echo "WARNING: Input directory not found: $INPUT_DIR"
        continue
    fi

    # Find all scene files
    SCENE_COUNT=0
    for scene_file in "$INPUT_DIR"/scene_*.json; do
        [ -e "$scene_file" ] || continue

        # Extract scene ID
        SCENE_ID=$(basename "$scene_file" | sed 's/scene_\([0-9]*\)\.json/\1/')

        # Find the actual scene base directory (handles nested vs flat structure)
        SCENE_BASE_DIR=$(find_scene_base_dir "$OUTPUT_DIR" "$INPUT_NAME" "$SCENE_ID")
        if [ -z "$SCENE_BASE_DIR" ]; then
            SKIPPED_NO_EVAL+=("${OUTPUT_NAME}:scene_${SCENE_ID}")
            continue
        fi

        # Check prerequisite: eval_result.json must exist
        EVAL_FILE="${SCENE_BASE_DIR}/scene_${SCENE_ID}/eval_result.json"
        if [ ! -f "$EVAL_FILE" ]; then
            SKIPPED_NO_EVAL+=("${OUTPUT_NAME}:scene_${SCENE_ID}")
            continue
        fi

        # Check skip condition: eval_result_v2.json already exists
        V2_FILE="${SCENE_BASE_DIR}/scene_${SCENE_ID}/eval_result_v2.json"
        if [ "$SKIP_EXISTING" = true ] && [ -f "$V2_FILE" ]; then
            SKIPPED_V2_EXISTS+=("${OUTPUT_NAME}:scene_${SCENE_ID}")
            continue
        fi

        # Add to processing list: input_name:output_name:scene_id
        ALL_SCENES+=("${INPUT_NAME}:${OUTPUT_NAME}:${SCENE_ID}")
        ((SCENE_COUNT++)) || true
    done

    echo "  $OUTPUT_NAME: found $SCENE_COUNT scenes to process"
done

echo ""
echo "Summary:"
echo "  Scenes to process: ${#ALL_SCENES[@]}"
echo "  Skipped (no eval_result.json): ${#SKIPPED_NO_EVAL[@]}"
if [ "$SKIP_EXISTING" = true ]; then
    echo "  Skipped (eval_result_v2.json exists): ${#SKIPPED_V2_EXISTS[@]}"
fi
echo ""

# Show skipped scenes if not too many
if [ ${#SKIPPED_NO_EVAL[@]} -gt 0 ] && [ ${#SKIPPED_NO_EVAL[@]} -le 20 ]; then
    echo "Scenes skipped (no eval_result.json):"
    for scene in "${SKIPPED_NO_EVAL[@]}"; do
        echo "  - $scene"
    done
    echo ""
elif [ ${#SKIPPED_NO_EVAL[@]} -gt 20 ]; then
    echo "Scenes skipped (no eval_result.json): ${#SKIPPED_NO_EVAL[@]} scenes (too many to list)"
    echo ""
fi

# Check if there's work to do
if [ ${#ALL_SCENES[@]} -eq 0 ]; then
    echo "No scenes to process. Exiting."
    exit 0
fi

# Dry run: just show what would be processed
if [ "$DRY_RUN" = true ]; then
    echo "========================================"
    echo "DRY RUN - Would process ${#ALL_SCENES[@]} scenes:"
    echo "========================================"

    # Group by method for cleaner output
    declare -A METHOD_SCENES
    for scene_spec in "${ALL_SCENES[@]}"; do
        IFS=':' read -r INPUT_NAME OUTPUT_NAME SCENE_ID <<< "$scene_spec"
        METHOD_SCENES[$OUTPUT_NAME]+="${SCENE_ID},"
    done

    for method in "${!METHOD_SCENES[@]}"; do
        SCENES="${METHOD_SCENES[$method]}"
        SCENES="${SCENES%,}"  # Remove trailing comma
        SCENE_COUNT=$(echo "$SCENES" | tr ',' '\n' | wc -l)
        echo ""
        echo "$method ($SCENE_COUNT scenes):"
        echo "  Scene IDs: $SCENES"
    done

    echo ""
    echo "Would distribute across $NUM_WORKERS workers"
    SCENES_PER_WORKER=$(( (${#ALL_SCENES[@]} + NUM_WORKERS - 1) / NUM_WORKERS ))
    echo "Approx $SCENES_PER_WORKER scenes per worker"
    echo ""
    echo "To run for real, remove --dry-run flag"
    exit 0
fi

# Distribute scenes across workers
TOTAL_SCENES=${#ALL_SCENES[@]}
SCENES_PER_WORKER=$(( (TOTAL_SCENES + NUM_WORKERS - 1) / NUM_WORKERS ))

echo "Distributing $TOTAL_SCENES scenes across $NUM_WORKERS workers (~$SCENES_PER_WORKER per worker)"
echo ""

# Cleanup function to kill all workers
PIDS=()
cleanup() {
    echo ""
    echo "Caught signal, terminating workers..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
        fi
    done
    sleep 1
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null
        fi
    done
    echo "Workers terminated."
    exit 130
}

trap cleanup INT TERM EXIT

# Launch workers
WORKER_COUNT=0
for ((i=0; i<TOTAL_SCENES; i+=SCENES_PER_WORKER)); do
    # Get slice of scenes for this worker
    END=$((i + SCENES_PER_WORKER))
    if [ $END -gt $TOTAL_SCENES ]; then
        END=$TOTAL_SCENES
    fi

    # Build comma-separated scene specs for this worker
    WORKER_SPECS=""
    for ((j=i; j<END; j++)); do
        if [ -n "$WORKER_SPECS" ]; then
            WORKER_SPECS="$WORKER_SPECS,${ALL_SCENES[$j]}"
        else
            WORKER_SPECS="${ALL_SCENES[$j]}"
        fi
    done

    if [ -z "$WORKER_SPECS" ]; then
        continue
    fi

    WORKER_SCENE_COUNT=$((END - i))
    echo "Starting worker $WORKER_COUNT: $WORKER_SCENE_COUNT scenes"

    # Launch worker
    "${SCRIPT_DIR}/run_worker_recompute_vlm.sh" \
        "$WORKER_COUNT" \
        "$WORKER_SPECS" \
        "$STATE_DIR" \
        "$MAX_RETRIES" \
        > "${LOG_DIR}/worker_${WORKER_COUNT}.log" 2>&1 &

    PIDS+=($!)
    WORKER_COUNT=$((WORKER_COUNT + 1))
done

echo ""
echo "All $WORKER_COUNT workers launched. PIDs: ${PIDS[*]}"
echo "Logs: ${LOG_DIR}/worker_*.log"
echo ""
echo "Waiting for completion..."
echo "(Use 'tail -f ${LOG_DIR}/worker_0.log' to monitor progress)"
echo ""

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

# Summary by method
echo "Summary by method:"
declare -A METHOD_COMPLETED
declare -A METHOD_FAILED
for state_file in "$STATE_DIR"/worker_*_state.json; do
    if [ -f "$state_file" ]; then
        # Count completed per method
        while IFS= read -r scene_spec; do
            [ -n "$scene_spec" ] || continue
            IFS=':' read -r _ OUTPUT_NAME _ <<< "$scene_spec"
            METHOD_COMPLETED[$OUTPUT_NAME]=$((${METHOD_COMPLETED[$OUTPUT_NAME]:-0} + 1))
        done < <(jq -r '.completed[]' "$state_file" 2>/dev/null)

        # Count failed per method
        while IFS= read -r scene_spec; do
            [ -n "$scene_spec" ] || continue
            IFS=':' read -r _ OUTPUT_NAME _ <<< "$scene_spec"
            METHOD_FAILED[$OUTPUT_NAME]=$((${METHOD_FAILED[$OUTPUT_NAME]:-0} + 1))
        done < <(jq -r '.failed[].scene' "$state_file" 2>/dev/null)
    fi
done

for method in "${!METHOD_COMPLETED[@]}"; do
    completed=${METHOD_COMPLETED[$method]:-0}
    failed=${METHOD_FAILED[$method]:-0}
    echo "  $method: $completed completed, $failed failed"
done
echo ""

if [ $FAILED -eq 1 ]; then
    echo "SOME WORKERS HAD FAILURES"
    echo ""
    echo "Failed scenes:"
    for state_file in "$STATE_DIR"/worker_*_state.json; do
        if [ -f "$state_file" ]; then
            jq -r '.failed[] | "  - \(.scene) (after \(.attempts) attempts)"' "$state_file" 2>/dev/null
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

# Verification hint
echo ""
echo "Verify results with:"
echo "  ls ${OUTPUT_PARENT}/<method>/scene_*/eval_result_v2.json | wc -l"
echo "  jq '.results | keys' ${OUTPUT_PARENT}/<method>/scene_0/eval_result_v2.json"
