#!/bin/bash
# Download HSSD decomposed parts from HuggingFace using git sparse checkout
# This avoids rate limits by using git-lfs instead of individual file requests
#
# Usage: ./scripts/download_hssd_decomposed.sh [target_dir]
#
# Arguments:
#   target_dir  Directory to download to (default: _data/hssd)

set -e

TARGET_DIR="${1:-_data/hssd}"
REPO_URL="https://huggingface.co/datasets/hssd/hssd-hab"
TEMP_REPO="$TARGET_DIR/.hssd-hab-temp"

echo "========================================"
echo "HSSD Decomposed Parts Downloader"
echo "========================================"
echo "Target directory: $TARGET_DIR"
echo "Repository: $REPO_URL"
echo "========================================"
echo ""

# Check for git-lfs
if ! command -v git-lfs &> /dev/null; then
    echo "Error: git-lfs is not installed"
    echo "Install with: sudo apt install git-lfs"
    exit 1
fi

# Create target directory
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# Clean up any previous failed attempt
if [ -d "$TEMP_REPO" ]; then
    echo "Cleaning up previous temp directory..."
    rm -rf "$TEMP_REPO"
fi

echo "Step 1/5: Initializing git-lfs..."
git lfs install

echo ""
echo "Step 2/5: Cloning repository (metadata only)..."
GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --sparse "$REPO_URL" "$TEMP_REPO"

cd "$TEMP_REPO"

echo ""
echo "Step 3/5: Configuring sparse checkout for decomposed parts..."
git sparse-checkout set objects/decomposed

echo ""
echo "Step 4/5: Downloading decomposed GLB files via git-lfs..."
echo "This may take a while depending on your connection..."
git lfs pull --include="objects/decomposed/**/*_part_*.glb"

echo ""
echo "Step 5/5: Moving files to target directory..."
if [ -d "objects/decomposed" ]; then
    # Create objects dir if needed and move decomposed folder
    mkdir -p "../objects"
    if [ -d "../objects/decomposed" ]; then
        echo "Merging with existing decomposed directory..."
        cp -r objects/decomposed/* ../objects/decomposed/
    else
        mv objects/decomposed ../objects/
    fi
    echo "Files moved to: $TARGET_DIR/objects/decomposed/"
else
    echo "Warning: No decomposed directory found in clone"
fi

echo ""
echo "Cleaning up temporary repository..."
cd ..
rm -rf "$TEMP_REPO"

echo ""
echo "========================================"
echo "Download complete!"
echo ""
echo "Decomposed parts location: $TARGET_DIR/objects/decomposed/"
if [ -d "objects/decomposed" ]; then
    PART_COUNT=$(find objects/decomposed -name "*_part_*.glb" 2>/dev/null | wc -l)
    echo "Total part files: $PART_COUNT"
fi
echo "========================================"
