from msight_core.nodes import DataProcessingNode, NodeConfig
from msight_core.data import ImageData, DetectionResultsData
import yaml
from pathlib import Path
import numpy as np
import math
from .. import MergedDetector, HashLocalizer, ClassicWarperWithExternalUpdate
from msight_vision.base import DetectedObject2D, DetectionResult2D
import torch
import time
from msight_core.utils import get_redis_client
from PIL import Image as PILImage


device = "cuda" if torch.cuda.is_available() else "cpu"

# RF-DETR model size registry — mirrors CustomRFDETRObjectDetection.MODEL_REGISTRY from Project1.
# Populated lazily so rfdetr is not a hard import for users who only use YOLO nodes.
_RFDETR_MODEL_REGISTRY = None

def _get_rfdetr_registry():
    global _RFDETR_MODEL_REGISTRY
    if _RFDETR_MODEL_REGISTRY is None:
        try:
            import rfdetr_plus  # must precede rfdetr to avoid circular import for XLarge/2XLarge
            from rfdetr import (
                RFDETRNano, RFDETRSmall, RFDETRMedium,
                RFDETRLarge, RFDETRXLarge, RFDETR2XLarge,
            )
            _RFDETR_MODEL_REGISTRY = {
                "rfdetr_nano":    RFDETRNano,
                "rfdetr_small":   RFDETRSmall,
                "rfdetr_medium":  RFDETRMedium,
                "rfdetr_large":   RFDETRLarge,
                "rfdetr_xlarge":  RFDETRXLarge,
                "rfdetr_2xlarge": RFDETR2XLarge,
            }
        except ImportError as exc:
            raise ImportError(
                "rfdetr and rfdetr_plus packages are required for RFDETRDetectionNode. "
                "Install them with: pip install rfdetr rfdetr-plus"
            ) from exc
    return _RFDETR_MODEL_REGISTRY


def fisheye_ground_contact(
    bbox_xyxy: list,
    x0: float,
    y0: float,
) -> tuple:
    """Return the fisheye-corrected ground-contact pixel for a bounding box.

    In a fisheye image the projected "down" direction is radially away from the
    optical centre (x0, y0).  The ground-contact pixel is the intersection of the
    outward radial ray from (x0, y0) through the box centre with the box boundary
    (the farthest edge from the optical centre).

    Migrated from Project1 MSight/localize_dataset.py — FiftyOne dependencies removed.
    Falls back to axis-aligned bottom-centre when the box centre coincides with (x0, y0).
    """
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    dx, dy = cx - x0, cy - y0
    d = math.hypot(dx, dy)

    if d < 1e-6:
        return cx, y2

    ux, uy = dx / d, dy / d

    eps = 1e-9
    t_best = None

    if abs(ux) > eps:
        for xe in (x1, x2):
            t = (xe - x0) / ux
            if t > eps:
                py = y0 + t * uy
                if y1 - eps <= py <= y2 + eps:
                    if t_best is None or t < t_best:
                        t_best = t

    if abs(uy) > eps:
        for ye in (y1, y2):
            t = (ye - y0) / uy
            if t > eps:
                px = x0 + t * ux
                if x1 - eps <= px <= x2 + eps:
                    if t_best is None or t < t_best:
                        t_best = t

    if t_best is None:
        return cx, y2

    gx = max(x1, min(x2, x0 + t_best * ux))
    gy = max(y1, min(y2, y0 + t_best * uy))
    return gx, gy


def _sv_detections_to_detection_result2d(
    sv_detections,
    timestamp,
    sensor_type: str,
    class_names: list,
    detection_threshold: float,
    x0: float = None,
    y0: float = None,
) -> DetectionResult2D:
    """Convert supervision.Detections (RF-DETR output) to Project2 DetectionResult2D.

    When x0 and y0 (fisheye optical centre) are provided, pixel_bottom_center is set
    via fisheye_ground_contact() — the outward radial ray from (x0,y0) through the box
    centre intersected with the box boundary.  Without them, falls back to axis-aligned
    bottom-centre (cx, y2).

    Adapted from Project1 MSight/utils/fiftyone_to_msight_det.fo_detections_to_msight,
    removing all FiftyOne and WandB dependencies.
    """
    object_list = []
    if sv_detections is None or len(sv_detections) == 0:
        return DetectionResult2D(object_list=[], timestamp=timestamp, sensor_type=sensor_type)

    xyxy = sv_detections.xyxy          # shape (N, 4)
    confidences = sv_detections.confidence  # shape (N,) or None
    class_ids = sv_detections.class_id      # shape (N,) or None

    use_fisheye = x0 is not None and y0 is not None

    for i in range(len(sv_detections)):
        conf = float(confidences[i]) if confidences is not None else 1.0
        if conf < detection_threshold:
            continue

        x1, y1, x2, y2 = float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])
        cid = int(class_ids[i]) if class_ids is not None else 0

        if use_fisheye:
            cx, cy = fisheye_ground_contact([x1, y1, x2, y2], x0, y0)
        else:
            cx = (x1 + x2) / 2.0
            cy = y2

        obj = DetectedObject2D(
            box=[x1, y1, x2, y2],
            class_id=cid,
            score=conf,
            pixel_bottom_center=[cx, cy],
        )
        object_list.append(obj)

    return DetectionResult2D(object_list=object_list, timestamp=timestamp, sensor_type=sensor_type)


def load_locmaps(loc_maps_path):
    """
    Load localization maps from the specified path.
    :param loc_maps_path: path to the localization maps in the config file
    :return: localization maps
    """
    result = {key: np.load(item) for key, item in loc_maps_path.items()}
    return result


def _ensure_model_weights(model_path: str, config_dir: Path = None) -> Path:
    """Resolve model_path and download from HuggingFace if the file is missing.

    Downloads from mcity-ai/rfdetr_2xlarge (filename rfdetr_2xlarge_best.pt)
    when the configured weight file does not exist on disk.
    """
    path = Path(model_path)
    if not path.is_absolute() and config_dir is not None:
        path = config_dir / path
    path = path.resolve()

    if path.exists():
        return path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            f"Weight file not found at {path} and huggingface_hub is not installed. "
            "Install it with: pip install huggingface_hub"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Weight file not found at {path}. Downloading from HuggingFace mcity-ai/rfdetr_2xlarge ...")
    downloaded = hf_hub_download(
        repo_id="mcity-ai/rfdetr_2xlarge",
        filename="rfdetr_2xlarge_best.pt",
        local_dir=str(path.parent),
    )
    # hf_hub_download may save under the HF cache name; rename to expected path if needed.
    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != path:
        downloaded_path.rename(path)
    print(f"Downloaded weights to {path}")
    return path


def load_intrinsics(intrinsics_path: str, config_dir: Path = None) -> dict:
    """Load fisheye camera intrinsics from a JSON file.

    Expected keys (same format as Project1 MSight/utils/load_locamaps.load_intrinsics):
        f  – focal length in pixels
        x0 – fisheye centre x (pixel column of optical axis)
        y0 – fisheye centre y (pixel row of optical axis)

    Relative paths are resolved relative to config_dir when provided.
    """
    import json
    path = Path(intrinsics_path)
    if not path.is_absolute() and config_dir is not None:
        path = config_dir / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {path}")
    with open(path) as fh:
        data = json.load(fh)
    return {k: float(v) for k, v in data.items()}

class YoloOneStageDetectionNode(DataProcessingNode):
    default_configs = NodeConfig(
        publish_topic_data_type=DetectionResultsData
    )
    def __init__(self, configs, det_configs_path):
        super().__init__(configs)
        self.det_config_path = Path(det_configs_path)
        self.detector = None
        with open(self.det_config_path, "r") as f:
            self.det_config = yaml.safe_load(f)
        self.no_warp = self.det_config['warper_config']['no_warp']            
        self.logger.info(f"Initializing YoloOneStageDetectionNode with config: {self.det_config}")
        self.detector = MergedDetector(model_config=self.det_config['model_config'], device=device)
        loc_maps_path = self.det_config["loc_maps"]
        loc_maps = load_locmaps(loc_maps_path)
        self.localizers = {key: HashLocalizer(item['x_map'], item['y_map']) for key, item in loc_maps.items()}
        if not self.no_warp:
            self.warper = ClassicWarperWithExternalUpdate()
            self.warper_matrix_redis_prefix = self.det_config["warper_config"]["redis_prefix"]
        else:
            self.warper_matrix_redis_prefix = None
        self.include_sensor_data_in_result = self.det_config["det_config"].get("include_sensor_data_in_result", False)
        self.sensor_type = self.det_config["det_config"].get("sensor_type", "fisheye")

    def get_warp_matrix_from_redis(self, sensor_name):
        redis_client = get_redis_client()
        key = self.warper_matrix_redis_prefix + f":{sensor_name}"
        warp_matrix_str = redis_client.get(key)
        
        if warp_matrix_str is None:
            self.logger.warning(f"No warp matrix found in Redis for sensor: {sensor_name}")
            return None

        # Decode bytes to string if needed
        if isinstance(warp_matrix_str, bytes):
            warp_matrix_str = warp_matrix_str.decode()

        # Remove brackets and convert back to numpy array
        warp_matrix = np.array(eval(warp_matrix_str))
        
        # You may need to reshape it, for example to 3x3 if that's your matrix size
        warp_matrix = warp_matrix.reshape((3, 3))  # adjust the shape if your matrix is different
        
        return warp_matrix


    def process(self, data: ImageData):
        self.logger.info(f"Processing image data from sensor: {data.sensor_name}, frame: {data.frame_id}")
        start = time.time()
        image = data.to_ndarray()
        sensor_name = data.sensor_name
        frame_id = data.frame_id
        # cv2.imshow("image", image)
        # cv2.waitKey(1)
        # print(image.shape)
        timestamp = data.capture_timestamp
        if not self.no_warp:
            # print(f"Image shape before warping: {image.shape}")
            warping_matrix = self.get_warp_matrix_from_redis(sensor_name)
            image = self.warper.warp(image, warping_matrix)
            # print(f"Image shape after warping: {image.shape}")
        # cv2.imshow("image", image)
        # cv2.waitKey(1)
        # print(image.shape)
        result = self.detector.detect(image, timestamp, self.sensor_type, sensor_name)
        localizer = self.localizers[sensor_name]
        localizer.localize(result)
        self.logger.info(f"Detection completed in {time.time() - start:.2f} seconds for sensor: {sensor_name}")
        def is_number(val):
            return isinstance(val, (int, float, np.number)) and np.isfinite(val)
        result.object_list = [obj for obj in result.object_list if is_number(obj.lat) and is_number(obj.lon)]
        raw_sensor_data = None
        if self.include_sensor_data_in_result:
            raw_sensor_data = data
        detection_result_data = DetectionResultsData(result, sensor_frame_id=data.frame_id, capture_timestamp=data.capture_timestamp, creation_timestamp=time.time(), sensor_name=sensor_name, raw_sensor_data=raw_sensor_data)
        # print(f"Detection results: {detection_result_data.to_dict()}")
        return detection_result_data


# ---------------------------------------------------------------------------
# RF-DETR detection node
# Migrated from Project1: CustomRFDETRObjectDetection.inference() in
# workflows/auto_labeling.py.  FiftyOne, WandB, Hugging Face, supervision
# ByteTrack, and training/evaluation logic have been stripped.  The
# sv.Detections → DetectionResult2D conversion replaces the
# fo_detections_to_msight() helper from MSight/utils/fiftyone_to_msight_det.py.
# Localization reuses Project2's existing HashLocalizer unchanged.
# ---------------------------------------------------------------------------

class RFDETRDetectionNode(DataProcessingNode):
    """DataProcessingNode that runs an RF-DETR model on incoming ImageData frames.

    Config YAML keys (under rfdetr_config):
        model_name  – one of rfdetr_nano / rfdetr_small / rfdetr_medium /
                      rfdetr_large / rfdetr_xlarge / rfdetr_2xlarge
        model_path  – path to the trained best.pt checkpoint
        num_classes – number of output classes the model was trained with
        class_names – (optional) list of class-name strings, index == class_id
        detection_threshold – confidence threshold (default 0.2)
        sensor_type – sensor lens type forwarded to DetectionResult2D (default "fisheye")
        include_sensor_data_in_result – bool, attach raw ImageData to output (default false)

    Shared keys:
        intrinsics      – path to a JSON file with fisheye intrinsics (f, x0, y0)
        loc_maps        – path to a single .npz calibration file (x_map, y_map arrays)
        warper_config.no_warp       – bool
        warper_config.redis_prefix  – Redis key prefix for warp matrices
    """

    default_configs = NodeConfig(
        publish_topic_data_type=DetectionResultsData
    )

    def __init__(self, configs, det_configs_path):
        super().__init__(configs)
        self.det_config_path = Path(det_configs_path)
        with open(self.det_config_path, "r") as f:
            self.det_config = yaml.safe_load(f)

        rfdetr_cfg = self.det_config["rfdetr_config"]
        self.detection_threshold = rfdetr_cfg.get("detection_threshold", 0.2)
        self.class_names = rfdetr_cfg.get("class_names") or []
        self.sensor_type = rfdetr_cfg.get("sensor_type", "fisheye")
        self.include_sensor_data_in_result = rfdetr_cfg.get("include_sensor_data_in_result", False)

        config_dir = self.det_config_path.parent

        # Fisheye intrinsics — single JSON file, same format as Project1.
        intr = load_intrinsics(self.det_config["intrinsics"], config_dir=config_dir)
        self.x0, self.y0 = intr["x0"], intr["y0"]

        # Localization map — single .npz file for this camera.
        loc_map_path = Path(self.det_config["loc_maps"])
        if not loc_map_path.is_absolute():
            loc_map_path = config_dir / loc_map_path
        loc_map_data = np.load(loc_map_path)
        self.localizer = HashLocalizer(loc_map_data["x_map"], loc_map_data["y_map"])

        # Warper (same pattern as YoloOneStageDetectionNode)
        self.no_warp = self.det_config["warper_config"]["no_warp"]
        if not self.no_warp:
            self.warper = ClassicWarperWithExternalUpdate()
            self.warper_matrix_redis_prefix = self.det_config["warper_config"]["redis_prefix"]
        else:
            self.warper_matrix_redis_prefix = None

        # RF-DETR model — lazy-loaded registry so rfdetr is not a hard dependency
        # for users who only use YOLO nodes (mirrors Project1 MODEL_REGISTRY pattern)
        model_name = rfdetr_cfg["model_name"].lower()
        num_classes = rfdetr_cfg["num_classes"]
        model_path = _ensure_model_weights(rfdetr_cfg["model_path"], config_dir=config_dir)

        registry = _get_rfdetr_registry()
        if model_name not in registry:
            raise ValueError(
                f"Unsupported RF-DETR model '{model_name}'. "
                f"Available: {list(registry.keys())}"
            )

        self.logger.info(f"Loading RF-DETR model '{model_name}' from {model_path}")
        ModelClass = registry[model_name]
        self.model = ModelClass(
            pretrain_weights=str(model_path),
            num_classes=num_classes,
            accept_platform_model_license=True,
        )
        self.logger.info("RF-DETR model loaded successfully")

    def get_warp_matrix_from_redis(self, sensor_name):
        """Retrieve the homography warp matrix stored in Redis (same as YoloOneStageDetectionNode)."""
        redis_client = get_redis_client()
        key = self.warper_matrix_redis_prefix + f":{sensor_name}"
        warp_matrix_str = redis_client.get(key)

        if warp_matrix_str is None:
            self.logger.warning(f"No warp matrix found in Redis for sensor: {sensor_name}")
            return None

        if isinstance(warp_matrix_str, bytes):
            warp_matrix_str = warp_matrix_str.decode()

        warp_matrix = np.array(eval(warp_matrix_str)).reshape((3, 3))
        return warp_matrix

    def process(self, data: ImageData):
        self.logger.info(f"Processing image data from sensor: {data.sensor_name}, frame: {data.frame_id}")
        start = time.time()

        image_ndarray = data.to_ndarray()
        sensor_name = data.sensor_name
        timestamp = data.capture_timestamp

        # Optional warp correction (same pattern as YoloOneStageDetectionNode)
        if not self.no_warp:
            warping_matrix = self.get_warp_matrix_from_redis(sensor_name)
            image_ndarray = self.warper.warp(image_ndarray, warping_matrix)

        # RF-DETR expects a PIL Image (mirrors Project1 Image.open usage in inference())
        pil_image = PILImage.fromarray(image_ndarray)

        # Run RF-DETR inference → supervision.Detections
        sv_detections = self.model.predict(pil_image, threshold=self.detection_threshold)

        # Convert supervision.Detections → DetectionResult2D.
        # pixel_bottom_center uses fisheye_ground_contact() with this camera's (x0, y0).
        result = _sv_detections_to_detection_result2d(
            sv_detections,
            timestamp=timestamp,
            sensor_type=self.sensor_type,
            class_names=self.class_names,
            detection_threshold=self.detection_threshold,
            x0=self.x0,
            y0=self.y0,
        )

        # Localize: fills obj.lat / obj.lon via pixel_bottom_center lookup
        self.localizer.localize(result)

        self.logger.info(
            f"RF-DETR detection completed in {time.time() - start:.2f}s "
            f"for sensor: {sensor_name}, objects before filter: {len(result.object_list)}"
        )

        def is_number(val):
            return isinstance(val, (int, float, np.number)) and np.isfinite(val)

        result.object_list = [
            obj for obj in result.object_list
            if is_number(obj.lat) and is_number(obj.lon)
        ]

        raw_sensor_data = data if self.include_sensor_data_in_result else None
        detection_result_data = DetectionResultsData(
            result,
            sensor_frame_id=data.frame_id,
            capture_timestamp=data.capture_timestamp,
            creation_timestamp=time.time(),
            sensor_name=sensor_name,
            raw_sensor_data=raw_sensor_data,
        )
        return detection_result_data
