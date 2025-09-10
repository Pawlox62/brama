#!/usr/bin/env bash
set -euo pipefail
RTSP_URL="${1:-rtsp://192.168.51.93:554/Streaming/Channels/101}"
FPS="${2:-5}"
WIDTH="${3:-1280}"

ffmpeg -nostdin -rtsp_transport tcp -i "$RTSP_URL" \
  -an -r "$FPS" -vf "scale=${WIDTH}:-1,format=yuvj420p" -vcodec mjpeg -f image2pipe - \
| sudo docker run -i --rm myopenalpr:local -j -c eu -

