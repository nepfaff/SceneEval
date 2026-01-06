#!/bin/bash
# Render all methods in parallel (one process per method)
#
# Usage: ./scripts/render_all_methods.sh [resolution]
#
# Arguments:
#   resolution  Image width and height in pixels (default: 1024)
#
# Examples:
#   ./scripts/render_all_methods.sh           # Default 1024x1024
#   ./scripts/render_all_methods.sh 512       # 512x512 for faster testing
#   ./scripts/render_all_methods.sh 2048      # High resolution
#
# Output:
#   Each method renders to its configured output directory
#   Worker logs saved to logs/render_<timestamp>/

set -e

# Parse arguments
RESOLUTION=${1:-1024}

# ============================================
# METHOD CONFIGURATION
# ============================================
# Add/remove methods and their output directories here
# Format: "METHOD_NAME:OUTPUT_DIR"

METHODS=(
    "SceneWeaver:output_eval/render_sceneweaver"
    "SceneAgent:output_eval/render_sceneagent_tmp"
    "Holodeck:output_eval/render_holodeck"
    "HSM:output_eval/render_hsm"
    "HSM_hf:output_eval/render_hsm_hf"
    "LayoutVLM_Curated:output_eval/render_layoutvlm_curated"
    "LayoutVLM_Objaverse:output_eval/render_layoutvlm_objaverse"
    "IDesign:output_eval/render_idesign"
)

# Resolution is appended to output dirs: output_eval/render_sceneweaver_1024

# ============================================
# SETUP
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="logs/render_${RUN_ID}"
mkdir -p "$LOG_DIR"

PIDS=()
WORKER_METHODS=()

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
        method=${WORKER_METHODS[$i]}
        if kill -0 "$pid" 2>/dev/null; then
            echo "Terminating $method (PID $pid)..."
            kill "$pid" 2>/dev/null || true
        fi
    done

    # Wait briefly for graceful shutdown
    echo "Waiting for graceful shutdown..."
    sleep 3

    # Force kill any remaining processes
    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        method=${WORKER_METHODS[$i]}
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing $method (PID $pid)..."
            kill -9 "$pid" 2>/dev/null || true
        fi
    done

    # Also kill any orphaned blender processes from this session
    pkill -9 -f "blender.*--python" 2>/dev/null || true

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
echo "Multi-Method Parallel Rendering"
echo "========================================"
echo "Run ID: $RUN_ID"
echo "Resolution: ${RESOLUTION}x${RESOLUTION}"
echo "Log directory: $LOG_DIR"
echo ""
echo "Methods to render:"

VALID_METHODS=()
for entry in "${METHODS[@]}"; do
    METHOD="${entry%%:*}"
    OUTPUT_DIR="${entry##*:}"
    INPUT_DIR="input/$METHOD"

    if [ -d "$INPUT_DIR" ]; then
        SCENE_COUNT=$(ls "$INPUT_DIR"/scene_*.json 2>/dev/null | wc -l)
        echo "  - $METHOD: $SCENE_COUNT scenes -> ${OUTPUT_DIR}_${RESOLUTION}"
        VALID_METHODS+=("$entry")
    else
        echo "  - $METHOD: SKIPPED (no input directory)"
    fi
done

echo ""
echo "========================================"

if [ ${#VALID_METHODS[@]} -eq 0 ]; then
    echo "Error: No valid methods found with input directories"
    exit 1
fi

echo ""
read -p "Start rendering ${#VALID_METHODS[@]} methods? [y/N] " -n 1 -r
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

for entry in "${VALID_METHODS[@]}"; do
    METHOD="${entry%%:*}"
    OUTPUT_DIR="${entry##*:}"

    FULL_OUTPUT_DIR="${OUTPUT_DIR}_${RESOLUTION}"
    echo "Starting: $METHOD -> $FULL_OUTPUT_DIR"

    .venv/bin/python main.py \
        evaluation_plan=room_views_plan \
        "evaluation_plan.input_cfg.scene_methods=[$METHOD]" \
        'evaluation_plan.input_cfg.scene_mode=all' \
        "evaluation_plan.evaluation_cfg.output_dir=$FULL_OUTPUT_DIR" \
        "blender.resolution_x=$RESOLUTION" \
        "blender.resolution_y=$RESOLUTION" \
        > "${LOG_DIR}/${METHOD}.log" 2>&1 &

    PIDS+=($!)
    WORKER_METHODS+=("$METHOD")

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
FAILED_METHODS=()
COMPLETED_METHODS=()

for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    method=${WORKER_METHODS[$i]}

    if wait $pid; then
        echo "[DONE] $method completed successfully"
        COMPLETED_METHODS+=("$method")
    else
        echo "[FAIL] $method failed (exit code: $?)"
        FAILED=1
        FAILED_METHODS+=("$method")
    fi
done

# Disable EXIT trap for normal completion
trap - INT TERM

echo ""
echo "========================================"
echo "RENDERING COMPLETE"
echo "========================================"
echo "Completed: ${#COMPLETED_METHODS[@]}/${#PIDS[@]} methods"

if [ ${#COMPLETED_METHODS[@]} -gt 0 ]; then
    echo ""
    echo "Successful:"
    for method in "${COMPLETED_METHODS[@]}"; do
        echo "  - $method"
    done
fi

if [ $FAILED -eq 1 ]; then
    echo ""
    echo "Failed:"
    for method in "${FAILED_METHODS[@]}"; do
        echo "  - $method (check $LOG_DIR/${method}.log)"
    done
    echo ""
    echo "========================================"
    exit 1
fi

echo ""
echo "Logs: $LOG_DIR/"
echo "========================================"
