#!/bin/bash
# Parallel script for fixing articulated meshes using Blender
#
# Usage: ./scripts/run_fix_articulated_parallel.sh <num_workers> [--force] <method1> [method2] ...
#
# Example:
#   ./scripts/run_fix_articulated_parallel.sh 16 --force SceneAgent_Ours_Room SceneAgent_Ours_House
#   ./scripts/run_fix_articulated_parallel.sh 32 SceneAgent_Ours_Room  # without force

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse arguments
NUM_WORKERS=$1
shift

FORCE=""
if [[ "$1" == "--force" ]]; then
    FORCE="--force"
    shift
fi

METHODS=("$@")

if [[ ${#METHODS[@]} -eq 0 ]]; then
    echo "Usage: $0 <num_workers> [--force] <method1> [method2] ..."
    echo "Example: $0 16 --force SceneAgent_Ours_Room SceneAgent_Ours_House"
    exit 1
fi

# Base directory
BASE_DIR="/home/ubuntu/efs/nicholas/scene-agent-eval-scenes/SceneEval_converted"

# Create log directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/logs/fix_articulated_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "====================================="
echo "Parallel Articulated Mesh Fix"
echo "====================================="
echo "Workers: $NUM_WORKERS"
echo "Force: ${FORCE:-disabled}"
echo "Methods: ${METHODS[*]}"
echo "Log dir: $LOG_DIR"
echo ""

# Collect all scenes to process
ALL_SCENES=()
for method in "${METHODS[@]}"; do
    input_dir="$BASE_DIR/$method"
    if [[ ! -d "$input_dir" ]]; then
        echo "WARNING: Directory not found: $input_dir (skipping)"
        continue
    fi

    # Find all scene directories
    for scene_dir in "$input_dir"/scene_*; do
        if [[ -d "$scene_dir" ]]; then
            scene_name=$(basename "$scene_dir")
            ALL_SCENES+=("$method:$scene_name")
        fi
    done
done

TOTAL_SCENES=${#ALL_SCENES[@]}
echo "Total scenes to process: $TOTAL_SCENES"
echo ""

if [[ $TOTAL_SCENES -eq 0 ]]; then
    echo "No scenes found!"
    exit 1
fi

# Calculate scenes per worker
SCENES_PER_WORKER=$(( (TOTAL_SCENES + NUM_WORKERS - 1) / NUM_WORKERS ))

# Create worker script
WORKER_SCRIPT="$LOG_DIR/worker.sh"
cat > "$WORKER_SCRIPT" << 'WORKER_EOF'
#!/bin/bash
# Worker script for fixing articulated meshes
WORKER_ID=$1
SCENES_FILE=$2
FORCE=$3
LOG_FILE=$4
SCRIPT_DIR=$5

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[Worker $WORKER_ID] Starting..."

while IFS= read -r scene_spec; do
    [[ -z "$scene_spec" ]] && continue

    METHOD="${scene_spec%%:*}"
    SCENE="${scene_spec##*:}"

    echo "[Worker $WORKER_ID] Processing $METHOD/$SCENE"

    INPUT_DIR="/home/ubuntu/efs/nicholas/scene-agent-eval-scenes/SceneEval_converted/$METHOD"

    # Run Blender to fix this scene
    blender --background --python "$SCRIPT_DIR/fix_articulated_meshes_blender.py" -- \
        "$INPUT_DIR" --scene "$SCENE" $FORCE 2>&1 | \
        grep -E "(Merged|Fixed|Error|Warning|Applied scale)" || true

done < "$SCENES_FILE"

echo "[Worker $WORKER_ID] Done"
WORKER_EOF
chmod +x "$WORKER_SCRIPT"

# Distribute scenes to workers and launch
echo "Launching $NUM_WORKERS workers..."
PIDS=()

for ((i=0; i<NUM_WORKERS; i++)); do
    # Calculate scene range for this worker
    START=$((i * SCENES_PER_WORKER))
    END=$((START + SCENES_PER_WORKER))
    [[ $END -gt $TOTAL_SCENES ]] && END=$TOTAL_SCENES

    if [[ $START -ge $TOTAL_SCENES ]]; then
        break
    fi

    # Create scenes file for this worker
    SCENES_FILE="$LOG_DIR/worker_${i}_scenes.txt"
    for ((j=START; j<END; j++)); do
        echo "${ALL_SCENES[$j]}" >> "$SCENES_FILE"
    done

    WORKER_LOG="$LOG_DIR/worker_${i}.log"

    # Launch worker in background
    bash "$WORKER_SCRIPT" "$i" "$SCENES_FILE" "$FORCE" "$WORKER_LOG" "$SCRIPT_DIR" &
    PIDS+=($!)

    echo "  Worker $i: scenes $START-$((END-1)) (PID: ${PIDS[-1]})"
done

ACTUAL_WORKERS=${#PIDS[@]}
echo ""
echo "Launched $ACTUAL_WORKERS workers. Waiting for completion..."
echo "Monitor progress: tail -f $LOG_DIR/worker_*.log"
echo ""

# Wait for all workers
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        ((FAILED++))
    fi
done

echo ""
echo "====================================="
echo "All workers completed"
echo "Failed workers: $FAILED"
echo "====================================="

# Summary
echo ""
echo "Summary of fixed objects:"
grep -h "Fixed [0-9]* articulated" "$LOG_DIR"/worker_*.log 2>/dev/null | \
    awk '{sum += $2} END {print "Total fixed: " sum " articulated objects"}'

exit $FAILED
