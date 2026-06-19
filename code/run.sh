#!/bin/bash

set -e

MAIN_ARGS=()
RENDER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video|--tracking|--max-frames)
      MAIN_ARGS+=("$1" "$2")
      RENDER_ARGS+=("$1" "$2")
      shift 2
      ;;
    --device|--imgsz)
      MAIN_ARGS+=("$1" "$2")
      shift 2
      ;;
    --rotation|--fps|--start-frame|--output-dir|--output-video)
      RENDER_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

python main.py "${MAIN_ARGS[@]}" > salida.log 2>&1
python tools/render_tracking_visualization.py "${RENDER_ARGS[@]}" > salida_tracking.log 2>&1
