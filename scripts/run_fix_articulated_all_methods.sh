#!/bin/bash
# Run fix_articulated_meshes using Blender on all SceneAgent methods
#
# Usage: ./scripts/run_fix_articulated_all_methods.sh [--force]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse arguments
FORCE=""
if [[ "$1" == "--force" ]]; then
    FORCE="--force"
fi

# Base directory for converted scenes
BASE_DIR="/home/ubuntu/efs/nicholas/scene-agent-eval-scenes/SceneEval_converted"

# All SceneAgent methods
METHODS=(
    "SceneAgent_Ours_Room"
    "SceneAgent_Ours_House"
    "SceneAgent_NoCritic"
    "SceneAgent_MaxOneCritic"
    "SceneAgent_NoAgentMemory"
    "SceneAgent_NoAssetValidation"
    "SceneAgent_NoObserveScene"
    "SceneAgent_NoSpecializedTools"
    "SceneAgent_HSSD"
)

echo "====================================="
echo "Fix Articulated Meshes (Blender Mode)"
echo "====================================="
echo "Force mode: ${FORCE:-disabled}"
echo ""

# Run for each method
for method in "${METHODS[@]}"; do
    input_dir="$BASE_DIR/$method"

    if [[ ! -d "$input_dir" ]]; then
        echo "WARNING: Directory not found: $input_dir (skipping)"
        continue
    fi

    echo "Processing $method..."

    # Run blender in background mode with the fix script
    blender --background --python "$SCRIPT_DIR/fix_articulated_meshes_blender.py" -- "$input_dir" $FORCE 2>&1 | \
        grep -E "(Processing|Fixed|Merged|Total:|Error|Warning)" || true

    echo ""
done

echo "====================================="
echo "All methods processed"
echo "====================================="
