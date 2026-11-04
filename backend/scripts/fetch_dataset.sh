#!/usr/bin/env bash
# Downloads the Middlebury "templeRing" multi-view dataset (47 photos, 640x480) and converts it to JPEG,
# so the pipeline can be tested without the phone app. Output: backend/samples/templeRing/*.jpg
set -euo pipefail
cd "$(dirname "$0")/.."
out=samples/templeRing
mkdir -p samples
if [ ! -d "$out" ]; then
  tmp=$(mktemp -d)
  curl -fL -o "$tmp/templeRing.zip" https://vision.middlebury.edu/mview/data/data/templeRing.zip
  unzip -q "$tmp/templeRing.zip" -d "$tmp"
  mkdir -p "$out"
  docker run --rm -v "$tmp/templeRing:/in:ro" -v "$PWD/$out:/out" orbitize-backend python -c '
import pathlib
from PIL import Image
for i, p in enumerate(sorted(pathlib.Path("/in").glob("*.png")), 1):
    Image.open(p).convert("RGB").save(f"/out/img_{i:04d}.jpg", quality=95)
'
  rm -rf "$tmp"
fi
echo "$(ls "$out" | wc -l) photos in $out"
