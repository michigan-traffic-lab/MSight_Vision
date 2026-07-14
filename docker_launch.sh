#!/bin/bash
# Docker Compose launcher for the RTSP RF-DETR detection pipeline.
#
# Usage:
#   ./docker_launch_rtsp.sh                                       # uses .env defaults
#   ./docker_launch_rtsp.sh rtsp://35.0.1.70:9000/1              # override RTSP URL
#   ./docker_launch_rtsp.sh rtsp://35.0.1.70:9000/1 gs_mcity_1  # override URL + sensor
#
# Arguments:
#   $1  RTSP URL      (default: value from .env, else rtsp://35.0.1.70:9000/1)
#   $2  Sensor name   (default: value from .env, else gs_mcity_1)
#
# Flags:
#   -d        Start in detached mode (containers keep running after Ctrl-C)
#   --cpu     Use CPU-only Docker base image (no NVIDIA GPU required)
#   --build   Force rebuild of Docker images before starting
#   --down    Stop and remove containers, then exit
#   --logs    Tail logs of all containers after start (implies -d)
#
# Once running, open http://localhost:9010 in a browser to see detections.

set -euo pipefail

REPO_DIR="$(dirname "$(realpath "$0")")"
ENV_FILE="${REPO_DIR}/.env"

# ── Parse flags ─────────────────────────────────────────────────────────────────
USE_CPU=false
DO_BUILD=false
DO_DOWN=false
DETACH=false
TAIL_LOGS=false
POSITIONAL=()

for arg in "$@"; do
  case "$arg" in
    -d)       DETACH=true ;;
    --cpu)    USE_CPU=true ;;
    --build)  DO_BUILD=true ;;
    --down)   DO_DOWN=true ;;
    --logs)   DETACH=true; TAIL_LOGS=true ;;
    --*)      echo "Unknown flag: $arg" >&2; exit 1 ;;
    *)        POSITIONAL+=("$arg") ;;
  esac
done

# ── Positional args override env values ─────────────────────────────────────────
RTSP_URL="${POSITIONAL[0]:-${RTSP_URL:-rtsp://35.0.1.70:9000/1}}"
SENSOR_NAME="${POSITIONAL[1]:-${SENSOR_NAME:-gs_mcity_1}}"

export RTSP_URL
export SENSOR_NAME
# Ensure VIDEO_INPUT is not set — RTSP_URL takes priority but belt-and-suspenders
export VIDEO_INPUT=""

# ── Build compose command ────────────────────────────────────────────────────────
COMPOSE_CMD="docker compose"
[ -f "$ENV_FILE" ] && COMPOSE_CMD="$COMPOSE_CMD --env-file $ENV_FILE"
COMPOSE_CMD="$COMPOSE_CMD -f ${REPO_DIR}/docker-compose.yml"
[ "$USE_CPU" = true ] && COMPOSE_CMD="$COMPOSE_CMD -f ${REPO_DIR}/docker-compose.cpu.yml"

# ── Down mode ───────────────────────────────────────────────────────────────────
if [ "$DO_DOWN" = true ]; then
  echo "Stopping containers..."
  $COMPOSE_CMD down
  exit 0
fi

# ── Print config ─────────────────────────────────────────────────────────────────
echo "──────────────────────────────────────────────"
echo "  RTSP URL:    $RTSP_URL"
echo "  Sensor:      $SENSOR_NAME"
echo "  GPU:         $([ "$USE_CPU" = true ] && echo "disabled (CPU mode)" || echo "enabled")"
echo "  Viewer:      http://localhost:9010"
echo "──────────────────────────────────────────────"

# ── Build ────────────────────────────────────────────────────────────────────────
if [ "$DO_BUILD" = true ]; then
  echo "Building images..."
  $COMPOSE_CMD build
fi

# ── Start ────────────────────────────────────────────────────────────────────────
if [ "$DETACH" = true ]; then
  $COMPOSE_CMD up -d
  echo "Containers running in background."
  echo "  Logs:  docker compose logs -f"
  echo "  Stop:  ./docker_launch_rtsp.sh --down"
  if [ "$TAIL_LOGS" = true ]; then
    echo "Tailing logs (Ctrl-C detaches — containers keep running)..."
    $COMPOSE_CMD logs -f
  fi
else
  $COMPOSE_CMD up
fi
