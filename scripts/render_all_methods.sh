#!/bin/bash
# Render all methods in parallel (one process per method)
#
# Usage: ./scripts/render_all_methods.sh [resolution] [input_dir] [output_dir]
#
# Arguments:
#   resolution  Image width and height in pixels (default: 1536)
#   input_dir   Full path to input directory containing scene_*.json (optional)
#   output_dir  Full path to output directory (optional)
#
# When input_dir/output_dir are not provided, uses METHODS array config.
# When provided, renders the single method specified in METHODS array
# from/to the given paths directly.
#
# Examples:
#   ./scripts/render_all_methods.sh                    # Use METHODS array config
#   ./scripts/render_all_methods.sh 512                # 512x512, METHODS array config
#   ./scripts/render_all_methods.sh 1536 /path/to/input /path/to/output
#
# GPU Distribution:
#   If multiple GPUs are detected, methods are assigned GPUs in round-robin
#   fashion (at most one GPU per method). If there are more methods than GPUs,
#   GPUs are reused across methods.
#
# Output:
#   Each method renders to its configured output directory
#   Worker logs saved to logs/render_<timestamp>/

set -e

# Parse arguments
RESOLUTION=${1:-1536}
INPUT_DIR_OVERRIDE=${2:-}
OUTPUT_DIR_OVERRIDE=${3:-}

# Expand paths (handle ~)
if [ -n "$INPUT_DIR_OVERRIDE" ]; then
    INPUT_DIR_OVERRIDE=$(eval echo "$INPUT_DIR_OVERRIDE")
fi
if [ -n "$OUTPUT_DIR_OVERRIDE" ]; then
    OUTPUT_DIR_OVERRIDE=$(eval echo "$OUTPUT_DIR_OVERRIDE")
fi

# ============================================
# METHOD CONFIGURATION
# ============================================
# Add/remove methods and their output directories here
# Format: "METHOD_NAME:OUTPUT_DIR"

METHODS=(
    # "SceneWeaver:output_eval/render_sceneweaver"
    # "SceneAgent:output_eval/render_sceneagent"
    # "SceneAgent_NoCritic:output_eval/render_sceneagent_nocritic"
    # "SceneAgent_MaxOneCritic:output_eval/render_sceneagent_maxonecritic"
    # "SceneAgent_NoAssetValidation:output_eval/render_sceneagent_noassetvalidation"
    # "SceneAgent_NoObserveScene:output_eval/render_sceneagent_noobservescene"
    # "SceneAgent_Ours_Room:output_eval/render_sceneagent_ours_room"
    # "SceneAgent_Ours_House:output_eval/render_sceneagent_ours_house"
    # "SceneAgent_NoSpecializedTools:output_eval/render_sceneagent_nospecializedtools"
    # "SceneAgent_HSSD:output_eval/render_sceneagent_hssd"
    # "SceneAgent_NoSpecializedTools:output_eval/render_sceneagent_nospecializedtools"
    # "SceneAgent_HSSD:output_eval/render_sceneagent_hssd"
    # "Holodeck:output_eval/render_holodeck"
    # "HSM:output_eval/render_hsm"
    # "HSM_hf:output_eval/render_hsm_hf"
    "LayoutVLM_Curated:output_eval/render_layoutvlm_curated"
    # "LayoutVLM_Objaverse:output_eval/render_layoutvlm_objaverse"
    # "IDesign:output_eval/render_idesign"
)

# Resolution is appended to output dirs: output_eval/render_sceneweaver_1024

# ============================================
# GPU DETECTION
# ============================================

# Detect available GPUs (silently, output shown in validation section)
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -gt 0 ]; then
        GPU_IDS=($(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null))
    else
        GPU_COUNT=0
        GPU_IDS=()
    fi
else
    GPU_COUNT=0
    GPU_IDS=()
fi

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
if [ -n "$INPUT_DIR_OVERRIDE" ]; then
    echo "Input: $INPUT_DIR_OVERRIDE (override)"
    echo "Output: $OUTPUT_DIR_OVERRIDE (override)"
fi
if [ "$GPU_COUNT" -gt 0 ]; then
    echo "GPUs: $GPU_COUNT available (IDs: ${GPU_IDS[*]})"
else
    echo "GPUs: None (CPU rendering)"
fi
echo "Log directory: $LOG_DIR"
echo ""
echo "Methods to render:"

VALID_METHODS=()
PREVIEW_GPU_INDEX=0
for entry in "${METHODS[@]}"; do
    METHOD="${entry%%:*}"
    DEFAULT_OUTPUT="${entry##*:}"

    # Use overrides if provided, otherwise use defaults
    if [ -n "$INPUT_DIR_OVERRIDE" ]; then
        INPUT_DIR="$INPUT_DIR_OVERRIDE"
        OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
    else
        INPUT_DIR="input/$METHOD"
        OUTPUT_DIR="${DEFAULT_OUTPUT}_${RESOLUTION}"
    fi

    if [ -d "$INPUT_DIR" ]; then
        SCENE_COUNT=$(ls "$INPUT_DIR"/scene_*.json 2>/dev/null | wc -l)

        # Count completed scenes (those with all 9 room views)
        COMPLETED=0
        if [ -d "$OUTPUT_DIR/$METHOD" ]; then
            for scene_dir in "$OUTPUT_DIR/$METHOD"/scene_*/; do
                if [ -d "$scene_dir/room_views" ]; then
                    view_count=$(ls "$scene_dir/room_views/"room_*.png 2>/dev/null | wc -l)
                    if [ "$view_count" -eq 9 ]; then
                        COMPLETED=$((COMPLETED + 1))
                    fi
                fi
            done
        fi
        REMAINING=$((SCENE_COUNT - COMPLETED))

        if [ "$GPU_COUNT" -gt 0 ]; then
            PREVIEW_GPU="${GPU_IDS[$PREVIEW_GPU_INDEX]}"
            echo "  - $METHOD: $REMAINING/$SCENE_COUNT to render ($COMPLETED complete) -> $OUTPUT_DIR [GPU $PREVIEW_GPU]"
            PREVIEW_GPU_INDEX=$(( (PREVIEW_GPU_INDEX + 1) % GPU_COUNT ))
        else
            echo "  - $METHOD: $REMAINING/$SCENE_COUNT to render ($COMPLETED complete) -> $OUTPUT_DIR"
        fi
        VALID_METHODS+=("$entry")
    else
        echo "  - $METHOD: SKIPPED (no input directory: $INPUT_DIR)"
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

GPU_INDEX=0
for entry in "${VALID_METHODS[@]}"; do
    METHOD="${entry%%:*}"
    DEFAULT_OUTPUT="${entry##*:}"

    # Determine input/output paths
    if [ -n "$INPUT_DIR_OVERRIDE" ]; then
        INPUT_DIR="$INPUT_DIR_OVERRIDE"
        OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
        # Extract parent dir for root_dir (input_dir without the method folder name)
        INPUT_ROOT=$(dirname "$INPUT_DIR")
    else
        INPUT_DIR="input/$METHOD"
        OUTPUT_DIR="${DEFAULT_OUTPUT}_${RESOLUTION}"
        INPUT_ROOT="input"
    fi

    # Assign GPU in round-robin fashion if GPUs are available
    if [ "$GPU_COUNT" -gt 0 ]; then
        ASSIGNED_GPU="${GPU_IDS[$GPU_INDEX]}"
        GPU_ENV="CUDA_VISIBLE_DEVICES=$ASSIGNED_GPU"
        echo "Starting: $METHOD -> $OUTPUT_DIR (GPU $ASSIGNED_GPU)"
        GPU_INDEX=$(( (GPU_INDEX + 1) % GPU_COUNT ))
    else
        GPU_ENV=""
        echo "Starting: $METHOD -> $OUTPUT_DIR (CPU)"
    fi

    # Build asset path override for methods that store assets with input
    # These methods have per-scene assets in their input directory
    ASSET_OVERRIDE=""
    case "$METHOD" in
        SceneAgent*)
            ASSET_OVERRIDE="assets.scene_agent.dataset_root_path=${INPUT_DIR}"
            ;;
        SceneWeaver)
            ASSET_OVERRIDE="assets.sceneweaver.dataset_root_path=${INPUT_DIR} assets.sw.dataset_root_path=${INPUT_DIR}"
            ;;
        IDesign)
            ASSET_OVERRIDE="assets.idesign.dataset_root_path=${INPUT_DIR}"
            ;;
    esac

    env $GPU_ENV .venv/bin/python main.py \
        evaluation_plan=room_views_plan \
        "evaluation_plan.input_cfg.scene_methods=[$METHOD]" \
        'evaluation_plan.input_cfg.scene_mode=all' \
        "evaluation_plan.evaluation_cfg.output_dir=$OUTPUT_DIR" \
        "evaluation_plan.input_cfg.root_dir=$INPUT_ROOT" \
        $ASSET_OVERRIDE \
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
