# MSight Vision

MSight Vision is the perception module of the MSight ecosystem. It provides camera-based 2D detection, multi-camera fusion, tracking, and state estimation components for roadside intelligence deployments.

This repository is designed to work with MSight Core for distributed runtime orchestration and MSight Base for shared data abstractions.

## Overview

MSight Vision focuses on the camera perception stack used in intelligent transportation systems (ITS):

- Object detection from monocular and fisheye camera streams
- Geometric warping and localization to map coordinates
- Multi-camera fusion of detections into unified road-user observations
- Multi-object tracking over time
- Kinematic state estimation for downstream analytics and V2X messaging

The package exposes reusable Python components and ready-to-run CLI entry points for integration into full MSight pipelines.

## Key Components

The main public components include:

- YoloDetector and Yolo26Detector for one-stage object detection
- HashLocalizer for map-based localization
- ClassicWarper and ClassicWarperWithExternalUpdate for image warping
- FuserBase for multi-sensor fusion extension
- SortTracker for multi-object tracking
- FiniteDifferenceStateEstimator for motion-state estimation

## Installation

### Prerequisites

- Python 3.8+
- pip

### Install from source

```bash
git clone https://github.com/michigan-traffic-lab/MSight_Vision.git
cd MSight_Vision
pip install -e .
```

### Install from PyPI

```bash
pip install msight-vision
```

### Runtime dependencies

MSight Vision d
epends on common scientific and vision libraries, including:

- numpy
- opencv-python
- matplotlib
- geopy
- filterpy
- ultralytics

For complete system deployment, install MSight Core and MSight Base alongside this module.

## Command-Line Tools

This package provides executable entry points for key perception nodes:

- msight_launch_yolo_onestage_detection
- msight_launch_sort_tracker
- msight_launch_custom_fuser
- msight_launch_finite_difference_state_estimator
- msight_launch_2d_viewer
- msight_launch_road_user_list_viewer

Use the help flag to inspect arguments:

```bash
msight_launch_yolo_onestage_detection --help
msight_launch_sort_tracker --help
```

## Example Workflow

A fullstack reference example is provided in examples/fullstack, demonstrating a complete pipeline:

1. Load multi-camera images
2. Run YOLO detection
3. Localize detections to map coordinates
4. Fuse detections across sensors
5. Track road users
6. Estimate state variables
7. Visualize the resulting trajectories

To explore the example:

```bash
cd examples/fullstack
python main.py -c config.yaml
```

## Repository Structure

- cli: executable launch scripts for MSight Vision nodes
- msight_vision: core perception library
- examples: runnable demonstrations and sample configurations
- docker: container build files for CPU and local development

## Development Notes

- Keep model and calibration assets outside source control when possible.
- Validate detector and tracker configuration files before deployment.
- Use dedicated topics per sensor stream in production MSight Core pipelines.

## License

This project is licensed under the BSD 3-Clause License. See the LICENSE file for details.

## Contact

For questions, issues, and collaboration requests, please use the repository issue tracker:

https://github.com/michigan-traffic-lab/MSight_Vision

## Main Developers

- Rusheng Zhang
- Depu Meng
- Haoyu Han


