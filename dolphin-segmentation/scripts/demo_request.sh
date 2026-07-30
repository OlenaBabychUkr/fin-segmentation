#!/usr/bin/env bash
# ===========================================================================
# demo_request.sh — quick curl-based demo of the segmentation API
# ===========================================================================
#
# Usage:
#   bash scripts/demo_request.sh                  # downloads test image
#   bash scripts/demo_request.sh dolphin.jpg      # use your own image
#
# Requires: curl, base64 (coreutils), jq (optional, for pretty-print)
# ===========================================================================

set -euo pipefail

API_BASE="http://localhost:8000"
IMAGE_FILE="${1:-}"
TEST_IMAGE="test_dolphin.jpg"

# ── Download test image if none provided ────────────────────────────────────
if [[ -z "$IMAGE_FILE" ]]; then
    if [[ ! -f "$TEST_IMAGE" ]]; then
        echo "Downloading test dolphin image …"
        curl -sL \
            "https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Tursiops_truncatus_01.jpg/640px-Tursiops_truncatus_01.jpg" \
            -o "$TEST_IMAGE"
        echo "Saved to $TEST_IMAGE"
    fi
    IMAGE_FILE="$TEST_IMAGE"
fi

# ── Health check ────────────────────────────────────────────────────────────
echo ""
echo "=== GET /health ==="
curl -sf "${API_BASE}/health" | (command -v jq &>/dev/null && jq . || cat)
echo ""

# ── Predict via multipart upload ────────────────────────────────────────────
echo "=== POST /predict/upload  (file: ${IMAGE_FILE}) ==="
RESPONSE=$(curl -sf \
    -F "file=@${IMAGE_FILE};type=image/jpeg" \
    -F "conf_threshold=0.25" \
    -F "include_crop=true" \
    "${API_BASE}/predict/upload")

# Save full response
echo "$RESPONSE" > example_response.json
echo "Full response saved to example_response.json"

# Print summary (truncate crop_base64 fields)
echo ""
echo "=== Response summary ==="
echo "$RESPONSE" | python3 -c "
import json, sys
r = json.load(sys.stdin)
for fin in r.get('fins', []):
    if fin.get('crop_base64'):
        fin['crop_base64'] = fin['crop_base64'][:60] + '...[base64 PNG]'
print(json.dumps(r, indent=2))
" 2>/dev/null || echo "$RESPONSE"

echo ""
echo "⚠️  NOTE: pretrained COCO model — may not detect dolphin fins specifically."
echo "   Replace MODEL_PATH with custom checkpoint after Assignment 2/3."
