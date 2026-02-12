#!/bin/bash
# Script to download pre-compiled HEF models from GitHub Releases

OUTPUT_DIR="$(dirname "$0")/../models"
mkdir -p "$OUTPUT_DIR"

VARIANTS=("n" "s" "m" "l")
BASE_URL="https://github.com/DanielDubinsky/yolo26_hailo/releases/latest/download"

echo "Downloading YOLO26 variants (n, s, m, l) to $OUTPUT_DIR..."

for variant in "${VARIANTS[@]}"; do
    MODEL_NAME="yolo26${variant}.hef"
    MODEL_URL="${BASE_URL}/${MODEL_NAME}"
    OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}"

    echo "Downloading ${MODEL_NAME}..."
    if wget -q --show-progress -O "$OUTPUT_FILE" "$MODEL_URL"; then
        echo "✓ Successfully downloaded ${MODEL_NAME}"
    else
        echo "✗ Error downloading ${MODEL_NAME}"
        echo "  URL attempted: $MODEL_URL"
        # We don't exit here so it tries to download the others
    fi
done

echo "Download process complete."
