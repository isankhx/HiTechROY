#!/usr/bin/env bash
# Pack the project into a zip next to the project folder (code, docs, config; no flight
# records, venv, datasets or training runs).
#   ./tools/make_zip.sh            -> ../thermoswarm_<date>_vN.zip  (N = next free number for today)
#   ./tools/make_zip.sh test-ёлки  -> ../thermoswarm_<date>_test-ёлки.zip
#   ./tools/make_zip.sh --runs     -> same, plus the latest flight record runs/<last>/
#   ./tools/make_zip.sh --model    -> same, plus models/*.pt (trained weights)
set -euo pipefail
self="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$self")/../.."
proj=thermoswarm
[ -d "$proj" ] || { echo "run it from inside the $proj folder"; exit 1; }

label="" ; with_runs=0 ; with_model=0
for a in "$@"; do
  case "$a" in
    --runs) with_runs=1 ;;
    --model) with_model=1 ;;
    -h|--help) sed -n 2,8p "$self"; exit 0 ;;
    *) label="$a" ;;
  esac
done

day=$(date +%Y-%m-%d)
if [ -z "$label" ]; then
  n=1
  while [ -e "${proj}_${day}_v${n}.zip" ]; do n=$((n + 1)); done
  label="v${n}"
fi
out="${proj}_${day}_${label}.zip"
[ -e "$out" ] && { echo "$out already exists — choose another name"; exit 1; }

excl=(-x "$proj/runs/*" "$proj/.venv/*" "*/__pycache__/*" "*.pyc" "$proj/tests/last_*.log"
         "$proj/dataset/*" "$proj/runs_yolo/*" "$proj/*.pt" "$proj/.git/*")
[ "$with_model" = 1 ] || excl+=("$proj/models/*.pt")

zip -qr "$out" "$proj" "${excl[@]}"

if [ "$with_runs" = 1 ]; then
  last=$(ls -1td "$proj"/runs/*/ 2>/dev/null | head -1 || true)
  if [ -n "$last" ]; then
    zip -qr "$out" "${last%/}"
    echo "+ flight record ${last%/}"
  else
    echo "(no flight records in $proj/runs)"
  fi
fi

size=$(du -h "$out" | cut -f1)
files=$(unzip -l "$out" | tail -1 | awk '{print $2}')
echo "$(pwd)/$out  ($size, $files files)"
