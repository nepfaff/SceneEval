#!/bin/bash
# Fault-tolerant worker wrapper that processes scenes one-by-one with retry logic
#
# This script wraps the scene evaluation to provide automatic recovery from segfaults
# and other crashes. Instead of passing all scenes to a single python process,
# it processes each scene individually with retry logic.
#
# Usage: run_worker_with_restart.sh <worker_id> <scene_list> <state_dir> <max_retries> <base_args...>
#
# Arguments:
#   worker_id   - Unique ID for this worker (used for state file naming)
#   scene_list  - Comma-separated list of scene IDs: "1,2,3,4,5"
#   state_dir   - Directory to store state files for progress tracking
#   max_retries - Maximum retry attempts per scene (default: 5)
#   base_args   - All remaining args are passed to python main.py
#
# State File:
#   The worker maintains a JSON state file at ${state_dir}/worker_${worker_id}_state.json
#   This tracks pending, completed, and failed scenes for recovery and debugging.

set -o pipefail

WORKER_ID=$1
SCENE_LIST=$2
STATE_DIR=$3
MAX_RETRIES=${4:-5}
shift 4
BASE_ARGS=("$@")

if [ -z "$WORKER_ID" ] || [ -z "$SCENE_LIST" ] || [ -z "$STATE_DIR" ]; then
    echo "Usage: $0 <worker_id> <scene_list> <state_dir> <max_retries> <base_args...>"
    echo "Example: $0 0 '1,2,3,4,5' ./logs/state 5 evaluation_plan=sceneagent_plan"
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
    # Parse comma-separated scene list into JSON array
    local scenes_json=$(echo "$SCENE_LIST" | tr ',' '\n' | jq -R . | jq -s .)
    cat > "$STATE_FILE" << EOF
{
  "worker_id": "$WORKER_ID",
  "started_at": "$(date -Iseconds)",
  "pending": $scenes_json,
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

# Get next pending scene (returns empty string if none left)
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
# This allows for manual recovery: delete state file to restart from scratch,
# or edit it to retry specific scenes
if [ ! -f "$STATE_FILE" ]; then
    init_state
    echo "[Worker $WORKER_ID] Initialized state file: $STATE_FILE"
else
    echo "[Worker $WORKER_ID] Resuming from existing state file: $STATE_FILE"
fi

echo "[Worker $WORKER_ID] Starting with scenes: $SCENE_LIST"
echo "[Worker $WORKER_ID] Max retries per scene: $MAX_RETRIES"
echo "[Worker $WORKER_ID] Base args: ${BASE_ARGS[*]}"
echo ""

# Track retry attempts per scene (in case of requeue)
declare -A ATTEMPT_COUNT

# Process scenes one at a time
SCENE=$(get_next_scene)
while [ -n "$SCENE" ]; do
    # Initialize or increment attempt count
    if [ -z "${ATTEMPT_COUNT[$SCENE]}" ]; then
        ATTEMPT_COUNT[$SCENE]=1
    else
        ATTEMPT_COUNT[$SCENE]=$((ATTEMPT_COUNT[$SCENE] + 1))
    fi
    ATTEMPT=${ATTEMPT_COUNT[$SCENE]}

    echo "========================================"
    echo "[Worker $WORKER_ID] Processing scene $SCENE (attempt $ATTEMPT/$MAX_RETRIES)"
    echo "========================================"

    mark_in_progress "$SCENE"

    # Run single scene evaluation
    # We use 'set +e' temporarily to capture the exit code without triggering script exit
    set +e
    python main.py \
        "${BASE_ARGS[@]}" \
        'evaluation_plan.input_cfg.scene_mode=list' \
        "evaluation_plan.input_cfg.scene_list=[$SCENE]"
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[Worker $WORKER_ID] Scene $SCENE completed successfully"
        mark_completed
    else
        echo "[Worker $WORKER_ID] Scene $SCENE failed with exit code $EXIT_CODE (attempt $ATTEMPT/$MAX_RETRIES)"

        if [ $ATTEMPT -lt $MAX_RETRIES ]; then
            echo "[Worker $WORKER_ID] Re-queuing scene $SCENE for retry..."
            requeue_scene
            # Brief pause before retry to avoid hammering in case of transient issues
            sleep 2
        else
            echo "[Worker $WORKER_ID] Scene $SCENE failed after $MAX_RETRIES attempts, marking as failed"
            mark_failed $ATTEMPT
        fi
    fi

    echo ""

    # Get next scene
    SCENE=$(get_next_scene)
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
    jq -r '.failed[] | "  - Scene \(.scene) (after \(.attempts) attempts)"' "$STATE_FILE"
    echo ""
    echo "[Worker $WORKER_ID] State file preserved for debugging: $STATE_FILE"
    exit 1
fi

echo "[Worker $WORKER_ID] All scenes completed successfully"
exit 0
