#!/bin/bash

# --- Best Practices for Error Handling ---
# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when performing parameter expansion.
set -u
# If any command in a pipeline fails, the whole pipeline fails.
set -o pipefail

motif_id=$1
# Use a full path for the logos directory
LOGOS_DIR="./motif_logos"
TRANSFAC_FILE="${LOGOS_DIR}/${motif_id}.transfac"
OUTPUT_IMG="${LOGOS_DIR}/${motif_id}.png"
JASPAR_URL="https://jaspar.elixir.no/api/v1/matrix/${motif_id}.transfac"

echo "Creating Logo for motif ${motif_id}"

# 1. Create directory safely. -p prevents error if it already exists.
mkdir -p "$LOGOS_DIR"

# Check if motif_id argument was provided
if [ -z "$motif_id" ]; then
    echo "Error: No motif ID provided." >&2
    echo "Usage: $0 <JASPAR_MOTIF_ID>" >&2
    exit 1
fi

# 2. Download the file and check for success immediately.
# The '||' operator executes the following command if the first command fails (non-zero exit status).
wget -O "$TRANSFAC_FILE" "$JASPAR_URL" || {
    echo "Error: Failed to download TRANSFAC file for motif ${motif_id}." >&2
    exit 1
}

echo "Download successful. Generating sequence logo..."

# 3. Generate the logo using full paths and specified flags
weblogo -f "$TRANSFAC_FILE" -D transfac -F png \
    --xlabel "" --ylabel "bits" \
    --fontsize 14 --number-fontsize 12 \
    -c classic \
    --ticmarks 1.0 \
    --number-interval 1 \
    --size large \
    --fineprint "" \
    --errorbars NO \
    -o "$OUTPUT_IMG" || {
    echo "Error: weblogo command failed." >&2
    exit 1
}

echo "Successfully created logo: ${OUTPUT_IMG}"
