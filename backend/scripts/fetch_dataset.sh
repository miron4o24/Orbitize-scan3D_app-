#!/usr/bin/env bash
# Изтегля данните "templeRing" на Middlebury (47 снимки, 640x480) и ги преобразува в JPEG,
# за да може обработката да се тества без приложението. Резултат: backend/samples/templeRing/*.jpg
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
