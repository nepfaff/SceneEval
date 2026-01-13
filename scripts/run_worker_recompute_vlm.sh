#!/bin/bash
# Worker script for VLM metric recomputation across mixed methods
#
# This script processes a batch of scenes from potentially different methods,
# running VLM metric recomputation for each scene individually with retry logic.
#
# Usage: run_worker_recompute_vlm.sh <worker_id> <scene_specs> <state_dir> <max_retries>
#
# Arguments:
#   worker_id   - Unique ID for this worker (used for state file naming)
#   scene_specs - Comma-separated list of "input_name:output_name:scene_id" specs
#   state_dir   - Directory to store state files for progress tracking
#   max_retries - Maximum retry attempts per scene (default: 3)
#
# State File:
#   The worker maintains a JSON state file at ${state_dir}/worker_${worker_id}_state.json
#   This tracks pending, completed, and failed scenes for recovery and debugging.

set -o pipefail

WORKER_ID=$1
SCENE_SPECS_STR=$2
STATE_DIR=$3
MAX_RETRIES=${4:-3}

# Fixed parent directories
INPUT_PARENT="/home/ubuntu/efs/nicholas/scene-agent-eval-scenes/SceneEval_converted"
OUTPUT_PARENT="/home/ubuntu/efs/nicholas/scene-agent-eval-scenes/SceneEval_results"

# Helper function to detect output structure and return correct output_dir for main.py
# Some methods have nested structure: OUTPUT_PARENT/OUTPUT_NAME/INPUT_NAME/scene_*/
# Others have flat structure: OUTPUT_PARENT/OUTPUT_NAME/scene_*/
# Returns the output_dir to pass to main.py so it writes to the correct location
get_output_dir_for_method() {
    local output_name="$1"
    local input_name="$2"
    local scene_id="$3"

    local output_dir="${OUTPUT_PARENT}/${output_name}"

    # Check if nested structure exists (e.g., LayoutVLM)
    if [ -d "${output_dir}/${input_name}/scene_${scene_id}" ]; then
        # Nested: main.py will create INPUT_NAME/scene_*/ inside output_dir
        echo "${output_dir}"
        return 0
    fi

    # Check if flat structure exists (most methods)
    if [ -d "${output_dir}/scene_${scene_id}" ]; then
        # Flat: main.py will create INPUT_NAME/scene_*/ but we want scene_*/ directly
        # So we pass OUTPUT_PARENT and let main.py create OUTPUT_NAME/scene_*/
        echo "${OUTPUT_PARENT}"
        return 0
    fi

    # Default to OUTPUT_PARENT (flat structure)
    echo "${OUTPUT_PARENT}"
    return 0
}

if [ -z "$WORKER_ID" ] || [ -z "$SCENE_SPECS_STR" ] || [ -z "$STATE_DIR" ]; then
    echo "Usage: $0 <worker_id> <scene_specs> <state_dir> <max_retries>"
    echo "Example: $0 0 'SceneAgent:SceneAgent:1,LayoutVLM:LayoutVLM:5' ./logs/state 3"
    exit 1
fi

STATE_FILE="${STATE_DIR}/worker_${WORKER_ID}_state.json"

# Check for jq dependency
if ! command -v jq &> /dev/null; then
    echo "[Worker $WORKER_ID] ERROR: jq is required but not installed."
    echo "Install with: sudo apt-get install jq"
    exit 1
fi

# Initialize state file with pending scenes
init_state() {
    # Parse comma-separated scene specs into JSON array
    local specs_json=$(echo "$SCENE_SPECS_STR" | tr ',' '\n' | jq -R . | jq -s .)
    cat > "$STATE_FILE" << EOF
{
  "worker_id": "$WORKER_ID",
  "started_at": "$(date -Iseconds)",
  "pending": $specs_json,
  "completed": [],
  "failed": [],
  "current": null
}
EOF
}

# Update state file atomically (write to tmp, then move)
update_state() {
    local tmp="${STATE_FILE}.tmp"
    if ! jq "$1" "$STATE_FILE" > "$tmp"; then
        echo "[Worker $WORKER_ID] ERROR: Failed to update state file"
        return 1
    fi
    mv "$tmp" "$STATE_FILE"
}

# Get next pending scene spec (returns empty string if none left)
get_next_scene() {
    jq -r '.pending[0] // empty' "$STATE_FILE"
}

# Mark scene as in-progress (removes from pending, sets as current)
mark_in_progress() {
    update_state ".current = \"$1\" | .pending = .pending[1:]"
}

# Mark current scene as completed
mark_completed() {
    update_state ".completed += [.current] | .current = null"
}

# Mark current scene as failed (with attempt count)
mark_failed() {
    local attempts=$1
    update_state ".failed += [{\"scene\": .current, \"attempts\": $attempts}] | .current = null"
}

# Re-queue current scene for retry (puts back at front of pending)
requeue_scene() {
    update_state ".pending = [.current] + .pending | .current = null"
}

# Update finished timestamp
mark_finished() {
    update_state ".finished_at = \"$(date -Iseconds)\""
}

# Initialize state file if it doesn't exist
if [ ! -f "$STATE_FILE" ]; then
    init_state
    echo "[Worker $WORKER_ID] Initialized state file: $STATE_FILE"
else
    echo "[Worker $WORKER_ID] Resuming from existing state file: $STATE_FILE"
fi

echo "[Worker $WORKER_ID] Starting VLM recomputation"
echo "[Worker $WORKER_ID] Input parent: $INPUT_PARENT"
echo "[Worker $WORKER_ID] Output parent: $OUTPUT_PARENT"
echo "[Worker $WORKER_ID] Max retries per scene: $MAX_RETRIES"
echo ""

# Track retry attempts per scene spec (in case of requeue)
declare -A ATTEMPT_COUNT

# Process scenes one at a time
SCENE_SPEC=$(get_next_scene)
while [ -n "$SCENE_SPEC" ]; do
    # Parse scene spec: input_name:output_name:scene_id
    IFS=':' read -r INPUT_NAME OUTPUT_NAME SCENE_ID <<< "$SCENE_SPEC"

    # Initialize or increment attempt count
    if [ -z "${ATTEMPT_COUNT[$SCENE_SPEC]}" ]; then
        ATTEMPT_COUNT[$SCENE_SPEC]=1
    else
        ATTEMPT_COUNT[$SCENE_SPEC]=$((ATTEMPT_COUNT[$SCENE_SPEC] + 1))
    fi
    ATTEMPT=${ATTEMPT_COUNT[$SCENE_SPEC]}

    echo "========================================"
    echo "[Worker $WORKER_ID] Processing ${OUTPUT_NAME}/scene_${SCENE_ID} (attempt $ATTEMPT/$MAX_RETRIES)"
    echo "========================================"

    mark_in_progress "$SCENE_SPEC"

    # Detect output structure and get correct output_dir
    EFFECTIVE_OUTPUT_DIR=$(get_output_dir_for_method "$OUTPUT_NAME" "$INPUT_NAME" "$SCENE_ID")

    # Run single scene VLM recomputation
    # Uses recompute_vlm_plan which has preserve_existing_metrics=True and output_suffix="v2"
    set +e
    python main.py \
        evaluation_plan=recompute_vlm_plan \
        "evaluation_plan.input_cfg.root_dir=${INPUT_PARENT}" \
        "evaluation_plan.input_cfg.scene_methods=[${INPUT_NAME}]" \
        "evaluation_plan.input_cfg.scene_mode=list" \
        "evaluation_plan.input_cfg.scene_list=[${SCENE_ID}]" \
        "evaluation_plan.evaluation_cfg.output_dir=${EFFECTIVE_OUTPUT_DIR}" \
        "assets.scene_agent.dataset_root_path=${INPUT_PARENT}/${INPUT_NAME}"
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[Worker $WORKER_ID] ${OUTPUT_NAME}/scene_${SCENE_ID} completed successfully"
        mark_completed
    else
        echo "[Worker $WORKER_ID] ${OUTPUT_NAME}/scene_${SCENE_ID} failed with exit code $EXIT_CODE (attempt $ATTEMPT/$MAX_RETRIES)"

        if [ $ATTEMPT -lt $MAX_RETRIES ]; then
            echo "[Worker $WORKER_ID] Re-queuing for retry..."
            requeue_scene
            sleep 2
        else
            echo "[Worker $WORKER_ID] Failed after $MAX_RETRIES attempts, marking as failed"
            mark_failed $ATTEMPT
        fi
    fi

    echo ""

    # Get next scene
    SCENE_SPEC=$(get_next_scene)
done

# Mark worker as finished
mark_finished

# Report final status
COMPLETED=$(jq '.completed | length' "$STATE_FILE")
FAILED=$(jq '.failed | length' "$STATE_FILE")

echo "========================================"
echo "[Worker $WORKER_ID] FINISHED"
echo "========================================"
echo "[Worker $WORKER_ID] Completed: $COMPLETED scenes"
echo "[Worker $WORKER_ID] Failed: $FAILED scenes"

if [ "$FAILED" -gt 0 ]; then
    echo "[Worker $WORKER_ID] Failed scenes:"
    jq -r '.failed[] | "  - \(.scene) (after \(.attempts) attempts)"' "$STATE_FILE"
    echo ""
    echo "[Worker $WORKER_ID] State file preserved for debugging: $STATE_FILE"
    exit 1
fi

echo "[Worker $WORKER_ID] All scenes completed successfully"
exit 0
