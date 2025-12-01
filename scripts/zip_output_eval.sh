#!/bin/bash
#
# zip_output_eval.sh - Create a selective zip archive of output_eval
#
# DESCRIPTION:
#   Creates a zip archive containing only .blend and .png files from the
#   output_eval directory. The full directory structure is preserved in
#   the archive, allowing proper extraction with all parent paths intact.
#
# USAGE:
#   ./zip_output_eval.sh [output_path]
#
# ARGUMENTS:
#   output_path  (optional) Path for the output zip file.
#                Defaults to /home/ubuntu/SceneEval/output_eval.zip
#
# EXAMPLES:
#   ./zip_output_eval.sh                          # Creates output_eval.zip
#   ./zip_output_eval.sh /tmp/my_archive.zip      # Creates custom output
#
# FILE TYPES INCLUDED:
#   - *.blend  (Blender scene files)
#   - *.png    (PNG image files)

SOURCE_DIR="/home/ubuntu/SceneEval/output_eval"
OUTPUT_ZIP="${1:-/home/ubuntu/SceneEval/output_eval.zip}"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory $SOURCE_DIR does not exist"
    exit 1
fi

cd /home/ubuntu/SceneEval
zip -r "$OUTPUT_ZIP" output_eval -i "*.png" -i "*.blend"

echo "Created: $OUTPUT_ZIP"
