#!/bin/bash
# Convert LayoutVLM scenes to SceneEval format in parallel
#
# Usage: ./scripts/convert_scenes_parallel_layoutvlm.sh SOURCE_DIR TARGET_DIR [NUM_WORKERS]
#
# Arguments:
#   SOURCE_DIR   Path to LayoutVLM results directory containing scene_* dirs
#   TARGET_DIR   Path to SceneEval input directory (will be created)
#   NUM_WORKERS  Number of parallel workers (default: 64)
#
# Examples:
#   ./scripts/convert_scenes_parallel_layoutvlm.sh \
#       ~/LayoutVLM/results \
#       input/LayoutVLM \
#       64
#
# Output:
#   Converted scenes saved to TARGET_DIR
#   Worker logs saved to logs/convert_layoutvlm_<timestamp>/

set -e

# ============================================
# PARSE ARGUMENTS
# ============================================

if [ $# -lt 2 ]; then
    echo "Usage: $0 SOURCE_DIR TARGET_DIR [NUM_WORKERS]"
    echo ""
    echo "Arguments:"
    echo "  SOURCE_DIR   Path to LayoutVLM results directory containing scene_* dirs"
    echo "  TARGET_DIR   Path to SceneEval input directory (will be created)"
    echo "  NUM_WORKERS  Number of parallel workers (default: 64)"
    exit 1
fi

SOURCE_DIR="$1"
TARGET_DIR="$2"
NUM_WORKERS=${3:-64}

# Expand paths
SOURCE_DIR=$(eval echo "$SOURCE_DIR")
TARGET_DIR=$(eval echo "$TARGET_DIR")

# ============================================
# SETUP
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="logs/convert_layoutvlm_${RUN_ID}"
mkdir -p "$LOG_DIR"

PIDS=()
WORKER_SCENES=()

# ============================================
# CLEANUP HANDLER
# ============================================

cleanup() {
    echo ""
    echo "========================================"
    echo "Caught signal, terminating all workers..."
    echo "========================================"

    # First, try graceful termination
    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        if kill -0 "$pid" 2>/dev/null; then
            echo "Terminating worker $i (PID $pid)..."
            kill "$pid" 2>/dev/null || true
        fi
    done

    # Wait briefly for graceful shutdown
    echo "Waiting for graceful shutdown..."
    sleep 3

    # Force kill any remaining processes
    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing worker $i (PID $pid)..."
            kill -9 "$pid" 2>/dev/null || true
        fi
    done

    echo ""
    echo "All workers terminated."
    echo "Partial logs available in: $LOG_DIR/"
    exit 130
}

# Trap signals
trap cleanup INT TERM

# ============================================
# VALIDATION
# ============================================

echo "========================================"
echo "Parallel LayoutVLM Conversion"
echo "========================================"
echo "Run ID: $RUN_ID"
echo ""

# Validate source directory
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory not found: $SOURCE_DIR"
    exit 1
fi

# Discover scenes
SCENE_IDS=()
for dir in "$SOURCE_DIR"/scene_*; do
    if [ -d "$dir" ]; then
        # Extract scene ID from directory name (e.g., scene_002 -> 2)
        basename=$(basename "$dir")
        scene_id=${basename#scene_}
        # Remove leading zeros
        scene_id=$((10#$scene_id))
        SCENE_IDS+=($scene_id)
    fi
done

# Sort scene IDs
IFS=$'\n' SCENE_IDS=($(sort -n <<<"${SCENE_IDS[*]}")); unset IFS

TOTAL_SCENES=${#SCENE_IDS[@]}

if [ $TOTAL_SCENES -eq 0 ]; then
    echo "Error: No scene directories found in $SOURCE_DIR"
    exit 1
fi

# Adjust workers if more than scenes
if [ $NUM_WORKERS -gt $TOTAL_SCENES ]; then
    NUM_WORKERS=$TOTAL_SCENES
    echo "Note: Reduced workers to $NUM_WORKERS (number of scenes)"
fi

echo "Source: $SOURCE_DIR"
echo "Target: $TARGET_DIR"
echo "Scenes: $TOTAL_SCENES"
echo "Workers: $NUM_WORKERS"
echo "Log directory: $LOG_DIR"
echo ""

# Calculate scenes per worker (ceiling division)
SCENES_PER_WORKER=$(( (TOTAL_SCENES + NUM_WORKERS - 1) / NUM_WORKERS ))

echo "Distribution: ~$SCENES_PER_WORKER scenes per worker"
echo ""
echo "========================================"
echo ""

read -p "Start conversion with $NUM_WORKERS workers? [y/N] " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# ============================================
# LAUNCH WORKERS
# ============================================

echo ""
echo "Launching workers..."
echo ""

# Create target directory
mkdir -p "$TARGET_DIR"

for (( w=0; w<NUM_WORKERS; w++ )); do
    # Calculate start and end indices for this worker
    START_IDX=$((w * SCENES_PER_WORKER))
    END_IDX=$(( (w + 1) * SCENES_PER_WORKER ))

    # Don't exceed total scenes
    if [ $END_IDX -gt $TOTAL_SCENES ]; then
        END_IDX=$TOTAL_SCENES
    fi

    # Skip if no scenes for this worker
    if [ $START_IDX -ge $TOTAL_SCENES ]; then
        continue
    fi

    # Build comma-separated list of scene IDs for this worker
    WORKER_SCENE_LIST=""
    for (( i=START_IDX; i<END_IDX; i++ )); do
        if [ -z "$WORKER_SCENE_LIST" ]; then
            WORKER_SCENE_LIST="${SCENE_IDS[$i]}"
        else
            WORKER_SCENE_LIST="${WORKER_SCENE_LIST},${SCENE_IDS[$i]}"
        fi
    done

    SCENE_COUNT=$((END_IDX - START_IDX))
    echo "Starting worker $w: $SCENE_COUNT scenes (IDs: ${SCENE_IDS[$START_IDX]}-${SCENE_IDS[$((END_IDX-1))]})"

    .venv/bin/python conversion/layoutvlm/convert_SceneEval.py \
        "$SOURCE_DIR" \
        "$TARGET_DIR" \
        --scenes "$WORKER_SCENE_LIST" \
        > "${LOG_DIR}/worker_${w}.log" 2>&1 &

    PIDS+=($!)
    WORKER_SCENES+=("$WORKER_SCENE_LIST")

    echo "  PID: ${PIDS[-1]}"
done

echo ""
echo "========================================"
echo "All ${#PIDS[@]} workers launched"
echo "PIDs: ${PIDS[*]}"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_DIR/*.log"
echo ""
echo "Press Ctrl+C to stop all workers"
echo "========================================"
echo ""

# ============================================
# WAIT FOR COMPLETION
# ============================================

FAILED=0
FAILED_WORKERS=()
COMPLETED_WORKERS=()

for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}

    if wait $pid; then
        echo "[DONE] Worker $i completed successfully"
        COMPLETED_WORKERS+=($i)
    else
        echo "[FAIL] Worker $i failed (exit code: $?)"
        FAILED=1
        FAILED_WORKERS+=($i)
    fi
done

# Disable trap for normal completion
trap - INT TERM

echo ""
echo "========================================"
echo "CONVERSION COMPLETE"
echo "========================================"
echo "Completed: ${#COMPLETED_WORKERS[@]}/${#PIDS[@]} workers"

if [ ${#COMPLETED_WORKERS[@]} -gt 0 ]; then
    echo ""
    echo "Successful workers: ${COMPLETED_WORKERS[*]}"
fi

if [ $FAILED -eq 1 ]; then
    echo ""
    echo "Failed workers:"
    for w in "${FAILED_WORKERS[@]}"; do
        echo "  - Worker $w (check $LOG_DIR/worker_${w}.log)"
    done
    echo ""
    echo "========================================"
    exit 1
fi

# Count converted scenes
CONVERTED_COUNT=$(ls "$TARGET_DIR"/scene_*.json 2>/dev/null | wc -l)
echo ""
echo "Converted scenes: $CONVERTED_COUNT"
echo "Output: $TARGET_DIR"
echo "Logs: $LOG_DIR/"
echo "========================================"
