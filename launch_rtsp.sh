#!/bin/bash
# RTSP detection pipeline launcher.
# Opens one gnome-terminal tab per node (Redis, RTSP Source, RF-DETR Detector, Web Viewer).
#
# Usage:
#   ./launch_rtsp.sh
#   ./launch_rtsp.sh rtsp://35.0.1.70:9000/1
#   ./launch_rtsp.sh rtsp://35.0.1.70:9000/1 gs_mcity_1
#
# Arguments:
#   $1  RTSP URL       (default: rtsp://35.0.1.70:9000/1)
#   $2  Sensor name    (default: gs_mcity_1)
#
# Once running, open http://localhost:9010 in a browser to see detections.

REPO_DIR="$(dirname "$(realpath "$0")")"
VENV="${REPO_DIR}/venv/bin/activate"
DET_CONFIGS="${REPO_DIR}/examples/rfdetr/rfdetr_config.yaml"

export MSIGHT_EDGE_DEVICE_NAME=mcity_edge

RTSP_URL="${1:-rtsp://35.0.1.70:9000/1}"
SENSOR_NAME="${2:-gs_mcity_1}"

echo "RTSP URL:    $RTSP_URL"
echo "Sensor name: $SENSOR_NAME"
echo "Viewer:      http://localhost:9010"
echo ""

# Clear stale node registrations left by any previous unclean exit.
redis-cli hdel MSIGHT:NODES rtsp_source rfdetr_detector detection_viewer > /dev/null 2>&1 || true

gnome-terminal \
  --tab --title="Redis" \
    --command="bash -c 'redis-server; exec bash'" \
  --tab --title="RTSP Source (${SENSOR_NAME})" \
    --command="bash -c 'source ${VENV} && msight_launch_rtsp \
      --name rtsp_source \
      --sensor-name ${SENSOR_NAME} \
      --publish-topic camera/${SENSOR_NAME} \
      --url ${RTSP_URL} \
      --rtsp-transport tcp; exec bash'" \
  --tab --title="RF-DETR Detector" \
    --command="bash -c 'source ${VENV} && msight_launch_rfdetr_detection \
      --name rfdetr_detector \
      --subscribe-topic camera/${SENSOR_NAME} \
      --publish-topic detection/${SENSOR_NAME} \
      --det-configs ${DET_CONFIGS}; exec bash'" \
  --tab --title="Web Viewer (port 9010)" \
    --command="bash -c 'source ${VENV} && msight_launch_web_viewer \
      --name detection_viewer \
      --subscribe-topic detection/${SENSOR_NAME} \
      --port 9010; exec bash'"
