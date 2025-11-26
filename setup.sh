#!/bin/bash
set -e

# SceneEval Setup Script
# Downloads all data for all supported methods

DATA_DIR="_data"
INPUT_DIR="input"
ANNOTATIONS_URL="https://github.com/3dlg-hcvc/SceneEval/releases/download/SceneEval-500_v250610/SceneEval-500_v250610.zip"

echo "=== SceneEval Setup ==="
echo ""

# Install dependencies first (required for scripts that import bpy)
echo ">>> Installing Python dependencies..."
uv sync
echo "    Done."
echo ""

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$INPUT_DIR"

# 1. Download annotations
echo ">>> Downloading SceneEval-500 annotations..."
curl -L -o "/tmp/SceneEval-500.zip" "$ANNOTATIONS_URL"
unzip -o "/tmp/SceneEval-500.zip" -d "/tmp/SceneEval-500"
cp /tmp/SceneEval-500/annotations.csv "$INPUT_DIR/"
rm -rf /tmp/SceneEval-500 /tmp/SceneEval-500.zip
echo "    Done."

# 2. Download Objathor assets (for Holodeck)
echo ""
echo ">>> Downloading Objathor assets (for Holodeck)..."
echo "    This may take a while (~50GB)..."
uv run python scripts/prepare_objathor.py --data-dir "$DATA_DIR" --num-processes 5 <<< "y"
echo "    Done."

# 3. Download HSSD assets (for HSM)
echo ""
echo ">>> Downloading HSSD assets (for HSM)..."
echo "    Prerequisites: gltf-transform and ktx must be installed"
echo "    This may take a while (~80GB)..."
if command -v gltf-transform &> /dev/null && command -v ktx &> /dev/null; then
    uv run python scripts/prepare_hssd.py --data-dir "$DATA_DIR" --clone-method https <<< "y"
    echo "    Done."
else
    echo "    SKIPPED: gltf-transform or ktx not found in PATH"
    echo "    Install them manually and run: python scripts/prepare_hssd.py"
fi

# 4. Download LayoutVLM assets
echo ""
echo ">>> Downloading LayoutVLM assets..."
LAYOUTVLM_GDRIVE_ID="1WGbj8gWn-f-BRwqPKfoY06budBzgM0pu"
mkdir -p "$DATA_DIR/layoutvlm-objathor"
uv run gdown "$LAYOUTVLM_GDRIVE_ID" -O "/tmp/layoutvlm-objathor.zip"
unzip -q "/tmp/layoutvlm-objathor.zip" -d "$DATA_DIR/layoutvlm-objathor"
# Flatten nested directory structure if present
if [ -d "$DATA_DIR/layoutvlm-objathor/test_asset_dir" ]; then
    mv "$DATA_DIR/layoutvlm-objathor/test_asset_dir/"* "$DATA_DIR/layoutvlm-objathor/"
    rmdir "$DATA_DIR/layoutvlm-objathor/test_asset_dir"
fi
rm -f "/tmp/layoutvlm-objathor.zip"
echo "    Done."

# 5. Manual downloads (print instructions)
echo ""
echo "=== Manual Downloads Required ==="
echo ""
echo "The following dataset requires manual download due to licensing:"
echo ""
echo "3D-FUTURE (for ATISS, DiffuScene, LayoutGPT, InstructScene):"
echo "   - Visit: https://tianchi.aliyun.com/dataset/98063"
echo "   - Download and extract to: $DATA_DIR/3D-FUTURE-model/"
echo ""
echo "=== Setup Complete ==="
