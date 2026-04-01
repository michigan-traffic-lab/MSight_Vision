from pathlib import Path
from typing import List, Type, Dict, Any, Optional, Set
from msight_vision.base import DetectionResult2D
from .base import ImageDetector2DBase
from .detector_yolo import YoloDetector, Yolo26Detector, Yolo26OBBDetector
from numpy import ndarray


# Registry of available detector classes
DETECTOR_REGISTRY: Dict[str, Type[ImageDetector2DBase]] = {
    "YoloDetector": YoloDetector,
    "Yolo26Detector": Yolo26Detector,
    "Yolo26OBBDetector": Yolo26OBBDetector,
}

class MergedDetector(ImageDetector2DBase):
    """Merges multiple 2D detectors, each responsible for detecting specific class IDs."""

    def __init__(self, model_config: List[Dict[str, Any]], device='cpu'):
        """
        Initialize the merged detector from model_config.

        :param model_config: list of detector configurations, each containing:
            - type: detector class name (must be in DETECTOR_REGISTRY)
            - class_ids: list of class IDs this detector is responsible for,
                         or omit / set to null / "all" to keep all the detection results of the detector.
            - params: dict of constructor parameters for the detector
        """
        super().__init__()
        self.detectors: List[ImageDetector2DBase] = []
        self.class_id_filters: List[Optional[Set[int]]] = []
        seen_class_ids: Dict[int, str] = {}

        for det_cfg in model_config:
            det_type = det_cfg["type"]
            det_cls = DETECTOR_REGISTRY.get(det_type)
            if det_cls is None:
                raise ValueError(
                    f"Unknown detector type '{det_type}'. Available: {list(DETECTOR_REGISTRY.keys())}"
                )

            params = dict(det_cfg.get("params", {}))
            params["device"] = device
            for key in ("ckpt_path", "model_path"):
                if key in params:
                    params["model_path"] = Path(params.pop(key))

            print(params)
            self.detectors.append(det_cls(**params))

            # class_ids handling and conflict check
            raw_ids = det_cfg.get("class_ids", None)
            if raw_ids is None or raw_ids == "all":
                if len(model_config) > 1:
                    raise ValueError(
                        f"Detector '{det_type}' is responsible for all classes, but there are multiple detectors. When using multiple detectors, each must specify explicit class_ids to avoid conflicts."
                    )
                self.class_id_filters.append(None)
            else:
                for cid in raw_ids:
                    if cid in seen_class_ids:
                        raise ValueError(
                            f"Class ID {cid} is claimed by both '{seen_class_ids[cid]}' and '{det_type}'. Each class ID must be assigned to exactly one detector."
                        )
                    seen_class_ids[cid] = det_type
                self.class_id_filters.append(set(raw_ids))

    def detect(self, image: ndarray, timestamp, sensor_type) -> DetectionResult2D:
        all_detected_objects = []

        for detector, allowed_ids in zip(self.detectors, self.class_id_filters):
            result = detector.detect(image, timestamp, sensor_type)

            for obj in result.object_list:
                if allowed_ids is None or obj.class_id in allowed_ids:
                    all_detected_objects.append(obj)

        return DetectionResult2D(
            all_detected_objects,
            timestamp,
            sensor_type,
        )