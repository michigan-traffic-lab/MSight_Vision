# MSight Vision

MSight Vision is the camera perception module of the MSight roadside intelligence ecosystem. It provides 2D object detection (YOLO and RF-DETR), fisheye localization, multi-camera fusion, multi-object tracking, and state estimation for intelligent transportation deployments.

MSight Base (shared data types) and MSight Core (distributed node runtime) are installed automatically as dependencies.

---

## Prerequisites

- Python 3.10+
- Redis server (`sudo apt install redis-server`)
- Git

For Docker deployment: Docker Engine with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (GPU) or Docker Engine only (CPU).

---

## Quick Start

Choose one of two deployment paths:

| Path | Best for |
|------|----------|
| [Local (venv)](#local-installation) | Development, debugging, direct access to logs |
| [Docker](#docker-deployment) | Reproducible deployments, no display required, browser-based viewer |

---

## Local Installation

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/mcity/Mcity_MSight_Vision.git
cd Mcity_MSight_Vision
python3 -m venv venv
source venv/bin/activate
```

### 2. Install

```bash
pip install -e .
```

This installs MSight Vision and all dependencies including MSight Base, MSight Core, RF-DETR, and Flask.

> **OpenCV note:** MSight Base pulls in `opencv-python-headless`, which conflicts with the GUI viewer node. Fix it after installation:
> ```bash
> pip uninstall -y opencv-python-headless
> pip install --force-reinstall opencv-python
> ```
> This is not required if you use the web viewer (`msight_launch_web_viewer`) or Docker Compose.

### 3. Place calibration files

```
examples/rfdetr/
├── calibration/
│   └── intrinsics.json        # fisheye camera intrinsics
├── locmaps/
│   └── locmap_<site>.npz      # localization map (x_map, y_map arrays)
├── models/                    # auto-created on first run (gitignored)
└── rfdetr_config.yaml
```

`intrinsics.json` format:
```json
{ "f": 1234.5, "x0": 960.0, "y0": 540.0 }
```

### 4. Run

```bash
# Single MP4 file
./launch.sh /path/to/video.mp4

# All MP4 files in a folder, played sequentially
./launch.sh /path/to/folder/
```

This opens four terminal tabs — Redis, Video Source, RF-DETR Detector, and Web Viewer. Once all tabs are running, open **http://localhost:9010** in a browser to see detections.

---

## Docker Deployment

No local Python environment needed. The viewer streams to a browser — no display server or `xhost` required.

The pipeline image is tagged **`michigantrafficlab/msight-vision:latest`** at build time.

### 1. Configure

```bash
cp env_sample .env
```

Edit `.env` — choose one source:

```bash
# Live RTSP stream (takes priority when set)
RTSP_URL=rtsp://10.0.1.70:9000/1

# OR recorded video file / folder
# VIDEO_INPUT=/path/to/video.mp4

# Sensor name — must match across all nodes
SENSOR_NAME=gs_mcity_1

MSIGHT_EDGE_DEVICE_NAME=mcity_edge
```

### 2. Build image

Builds `michigantrafficlab/msight-vision:latest` from `docker/Dockerfile-local`:

```bash
./docker_launch.sh --build -d
```

Or with Docker Compose directly:

```bash
docker compose --env-file .env build
```

To build with a version tag:

```bash
MSIGHT_VISION_TAG=0.1.0 docker compose --env-file .env build
```

### 3. Deploy

**RTSP stream (GPU) — recommended:**

```bash
# Start detached (Ctrl-C safe, containers keep running)
./docker_launch.sh -d

# Override URL and sensor inline
./docker_launch.sh rtsp://10.0.1.70:9001/1 gs_mcity_2 -d
```

**Video file / folder:**

```bash
VIDEO_INPUT=/path/to/video.mp4 docker compose --env-file .env up -d
```

**CPU-only (no NVIDIA GPU):**

```bash
./docker_launch.sh --cpu -d
```

### 4. View detections

Open **http://localhost:9010** — the annotated MJPEG stream appears once the detector starts processing frames.

### Manage running containers

```bash
# Follow logs (Ctrl-C detaches, containers keep running)
./docker_launch.sh --logs

# Or directly:
docker compose logs -f

# Stop and remove containers
./docker_launch.sh --down
```

### `docker_launch.sh` flag reference

| Flag | Effect |
|------|--------|
| *(none)* | Foreground — Ctrl-C stops all containers |
| `-d` | Detached — containers keep running after Ctrl-C |
| `--build` | Rebuild `michigantrafficlab/msight-vision` image first |
| `--logs` | Start detached, then tail logs |
| `--cpu` | Use CPU-only base image (no GPU required) |
| `--down` | Stop and remove all containers |

---

## Model Weights

RF-DETR weights are **not included** in this repository. On first run, if `examples/rfdetr/models/rfdetr_2xlarge_best.pt` is missing, it downloads automatically from HuggingFace:

- Repo: `mcity-ai/rfdetr_2xlarge`  
- File: `rfdetr_2xlarge_best.pt`

To download manually:
```bash
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('mcity-ai/rfdetr_2xlarge', 'rfdetr_2xlarge_best.pt',
                local_dir='examples/rfdetr/models')
"
```

In Docker, the weights are downloaded into `./examples/rfdetr/models/` on the host and reused on subsequent runs.

---

## Configuration

Edit `examples/rfdetr/rfdetr_config.yaml`:

| Key | Description |
|-----|-------------|
| `rfdetr_config.model_name` | Model size: `rfdetr_nano` … `rfdetr_2xlarge` |
| `rfdetr_config.model_path` | Path to weights (relative to config file) |
| `rfdetr_config.num_classes` | Number of classes the model was trained with |
| `rfdetr_config.class_names` | Human-readable class labels (index = class ID) |
| `rfdetr_config.detection_threshold` | Confidence threshold (default `0.2`) |
| `rfdetr_config.sensor_type` | Lens type forwarded to detection result (`fisheye`) |
| `intrinsics` | Path to `intrinsics.json` (relative to config file) |
| `loc_maps` | Path to `.npz` localization map (relative to config file) |

---

## Web Viewer

Both `launch.sh` and Docker Compose use `msight_launch_web_viewer` instead of the legacy `cv2.imshow`-based viewer. It serves an MJPEG stream over HTTP:

- **URL:** `http://<host>:9010`
- **No X11 / display server required**
- Supports multiple simultaneous browser connections
- Shows bounding boxes, bottom-center ground contact points, class IDs, and confidence scores

To run the web viewer standalone:
```bash
msight_launch_web_viewer \
  --name detection_viewer \
  --subscribe-topic detection/my_camera \
  --port 9010
```

---

## Pipeline Architecture

```
Video / RTSP source
        │  camera/<sensor>
        ▼
RF-DETR Detection Node
        │  detection/<sensor>
        ▼
Web Viewer  ──► http://localhost:9010
```

All nodes communicate via Redis pub/sub. Each tab or container runs one node independently.

| Tab / Service | Role |
|---------------|------|
| **Redis** | Message broker between nodes |
| **Video Source** | Reads video frames, publishes to `camera/<sensor>` |
| **RF-DETR Detector** | Subscribes to frames, runs detection, publishes to `detection/<sensor>` |
| **Web Viewer** | Subscribes to detections, streams annotated frames to browser on port 9010 |

---

## CLI Entry Points

| Command | Description |
|---------|-------------|
| `msight_launch_rfdetr_detection` | RF-DETR detection node |
| `msight_launch_yolo_onestage_detection` | YOLO detection node |
| `msight_launch_mp4_folder` | Folder video source (cycles through `.mp4` files) |
| `msight_launch_rtsp` | RTSP / single-file video source |
| `msight_launch_web_viewer` | Browser-based detection viewer (MJPEG, port configurable) |
| `msight_launch_2d_viewer` | Desktop viewer (`cv2.imshow`, requires display) |
| `msight_launch_sort_tracker` | Multi-object tracking node |
| `msight_launch_custom_fuser` | Multi-camera fusion node |
| `msight_launch_finite_difference_state_estimator` | Kinematic state estimation node |
| `msight_launch_road_user_list_viewer` | Road-user list desktop viewer |

Use `--help` on any command for argument details.

---

## Repository Structure

```
MSight_Vision/
├── launch.sh                    # Local launcher — file/folder source (gnome-terminal tabs)
├── launch_rtsp.sh               # Local launcher — RTSP stream (gnome-terminal tabs)
├── docker_launch.sh        # Docker launcher — RTSP stream (wraps docker compose)
├── docker-compose.yml           # Compose file — GPU, supports RTSP and file sources
├── docker-compose.cpu.yml       # Compose override — CPU-only (no GPU)
├── env_sample                   # Environment variable template (copy to .env)
├── pyproject.toml
├── cli/                         # Entry-point scripts
├── docker/
│   ├── Dockerfile               # GPU production build
│   ├── Dockerfile-cpu           # CPU production build
│   └── Dockerfile-local         # Editable-install build (used by Compose)
│                                #   → tags image as michigantrafficlab/msight-vision
├── msight_vision/
│   └── msight_core/             # Detection, tracking, fusion, state estimation, viewer nodes
└── examples/
    └── rfdetr/
        ├── rfdetr_config.yaml   # Pipeline configuration
        ├── calibration/         # intrinsics.json (fisheye intrinsics, user-provided)
        ├── locmaps/             # locmap_<site>.npz (user-provided)
        └── models/              # Model checkpoint (auto-downloaded, gitignored)
```

---

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Contact

Issues and collaboration: https://github.com/mcity/Mcity_MSight_Vision/issues

## Main Developers

- Rusheng Zhang
- Depu Meng
- Haoyu Han
