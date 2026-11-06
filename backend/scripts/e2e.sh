#!/usr/bin/env bash
# End-to-end test through the HTTP API, exactly like the app: create job, upload, start, poll, download.
# Usage: API_TOKEN=... [API_URL=http://localhost:8000] scripts/e2e.sh <folder of .jpg | video.mp4> [out dir]
set -euo pipefail
API=${API_URL:-http://localhost:8000}
H="Authorization: Bearer ${API_TOKEN:?set API_TOKEN}"
src=${1:?usage: e2e.sh <photo folder | video file> [out dir]}
out=${2:-e2e-out}
field() { python3 -c "import sys, json; print(json.load(sys.stdin).get('$1') or '')"; }

id=$(curl -fsS -X POST -H "$H" "$API/jobs" | field id)
echo "job $id"
if [ -f "$src" ]; then
  # Same 8 MB chunks as the app.
  size=$(stat -c %s "$src")
  chunk=$((8 * 1024 * 1024))
  for ((off = 0; off < size; off += chunk)); do
    tail -c +$((off + 1)) "$src" | head -c $chunk |
      curl -fsS -X PUT -H "$H" -H "Content-Type: application/octet-stream" --data-binary @- "$API/jobs/$id/video?offset=$off" >/dev/null
  done
  echo "uploaded video, $size bytes"
  start="{\"flat_base\": true, \"video_bytes\": $size}"
else
  for f in "$src"/*.jpg; do
    curl -fsS -X PUT -H "$H" -H "Content-Type: image/jpeg" --data-binary @"$f" "$API/jobs/$id/photos/$(basename "$f")" >/dev/null
  done
  echo "uploaded $(ls "$src"/*.jpg | wc -l) photos"
  start='{"flat_base": true}'
fi
curl -fsS -X POST -H "$H" -H "Content-Type: application/json" -d "$start" "$API/jobs/$id/start" >/dev/null

start=$(date +%s)
last=""
while :; do
  status=$(curl -fsS -H "$H" "$API/jobs/$id")
  state=$(echo "$status" | field state)
  [ "$state" != "$last" ] && echo "$(($(date +%s) - start))s  $state" && last=$state
  case $state in
    done) break ;;
    failed) echo "error: $(echo "$status" | field error)"; exit 1 ;;
  esac
  sleep 5
done

mkdir -p "$out"
for fmt in stl glb 3mf; do
  curl -fsS -H "$H" -o "$out/model.$fmt" "$API/jobs/$id/result.$fmt?longest_mm=100"
done
echo "$status" | python3 -c "import sys, json; print('report:', json.load(sys.stdin)['report'])"
ls -l "$out"
