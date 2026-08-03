from numpy import ndarray
import numpy as np
from msight_vision.base import DetectionResult2D, DetectedObject2D
from .base import ImageDetector2DBase
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, List
import cv2

class YoloDetector(ImageDetector2DBase):
    """YOLOv5 detector for 2D images."""

    def __init__(self, model_path: Path, device: str = "cpu", confthre: float = 0.25, nmsthre: float = 0.45, fp16: bool = False, class_agnostic_nms: bool = False, pre_mask_path: Dict[str, Path] = None, post_mask_path: Dict[str, Path] = None, id_mapping: Dict[int, int] = None):
        """
        Initialize the YOLO detector.
        :param model_path: path to the YOLO model
        :param device: device to run the model on (e.g., 'cpu', 'cuda')
        :param pre_mask_path: optional {sensor_name: path} of binary masks applied to
            the image before inference; pixels outside the mask are blacked out.
        :param post_mask_path: optional {sensor_name: path} of binary masks applied to
            the detections after inference; a detection is dropped unless its box
            overlaps the mask. Independent of ``pre_mask_path`` -- either, both, or
            neither may be given, and they may point at different files.
        :param id_mapping: optional dict mapping original class ids to new class ids,
            e.g. {0: 4, 1: 4}. Ids not present in the dict are kept unchanged.
        """
        super().__init__()
        self.model = YOLO(str(model_path))
        self.device = device
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.fp16 = fp16
        self.class_agnostic_nms = class_agnostic_nms
        self.id_mapping = {int(k): int(v) for k, v in id_mapping.items()} if id_mapping is not None else None
        self.pre_mask = {
            key: np.repeat((np.load(item).astype(bool).astype(np.uint8) * 255)[:, :, np.newaxis], 3, axis=2) for key, item in pre_mask_path.items()
        } if pre_mask_path is not None else None
        self.post_mask = {
            key: np.load(item).astype(bool).astype(np.uint8) for key, item in post_mask_path.items()
        } if post_mask_path is not None else None

    def apply_pre_mask(self, image: ndarray, sensor_name) -> ndarray:
        """
        Black out the pixels outside this sensor's pre mask.
        :param image: the image about to be fed to the model.
        :param sensor_name: name of the sensor the image came from.
        :return: the masked image, or the image unchanged if no pre mask applies.
        """
        if self.pre_mask is None or sensor_name not in self.pre_mask:
            return image
        if self.pre_mask[sensor_name].shape[:2] != image.shape[:2]:
            raise ValueError(
                f"Pre mask dimensions {self.pre_mask[sensor_name].shape[:2]} do not match image dimensions {image.shape[:2]}"
            )
        return cv2.bitwise_and(image, self.pre_mask[sensor_name])

    def apply_post_mask(self, detected_objects: List, sensor_name, image_shape) -> List:
        """
        Drop the detections whose box does not overlap this sensor's post mask.
        :param detected_objects: the detections returned by the model.
        :param sensor_name: name of the sensor the image came from.
        :param image_shape: shape of the image the detections were made on.
        :return: the kept detections, or all of them if no post mask applies.
        """
        if self.post_mask is None or sensor_name not in self.post_mask:
            return detected_objects
        mask = self.post_mask[sensor_name]
        if mask.shape[:2] != image_shape[:2]:
            raise ValueError(
                f"Post mask dimensions {mask.shape[:2]} do not match image dimensions {image_shape[:2]}"
            )
        return [obj for obj in detected_objects if self.box_overlaps_mask(obj.box, mask)]

    @staticmethod
    def box_overlaps_mask(box: List[float], mask: ndarray) -> bool:
        """
        Test whether a detection box shares at least one pixel with a mask. The box is
        rasterized inside its own bounding box, so the cost follows the box, not the frame.
        :param box: the four OBB vertices (8 values), or an axis-aligned [x1, y1, x2, y2].
        :param mask: single-channel mask, non-zero inside the kept region.
        :return: True if the box overlaps the mask.
        """
        if len(box) == 8:
            corners = np.asarray(box, dtype=np.float64).reshape(4, 2)
        else:
            x1, y1, x2, y2 = box
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)

        height, width = mask.shape[:2]
        x_min = max(int(np.floor(corners[:, 0].min())), 0)
        y_min = max(int(np.floor(corners[:, 1].min())), 0)
        x_max = min(int(np.ceil(corners[:, 0].max())) + 1, width)
        y_max = min(int(np.ceil(corners[:, 1].max())) + 1, height)
        if x_min >= x_max or y_min >= y_max:
            return False                       # box lies entirely off-image

        polygon = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
        cv2.fillConvexPoly(polygon, np.round(corners - [x_min, y_min]).astype(np.int32), 1)
        return bool(np.any(mask[y_min:y_max, x_min:x_max] & polygon))

    def map_class_id(self, class_id: int) -> int:
        """
        Remap a class id according to ``id_mapping``.
        :param class_id: the original class id predicted by the model.
        :return: the remapped class id, or the original id if it is not in ``id_mapping``.
        """
        if self.id_mapping is not None:
            return self.id_mapping.get(class_id, class_id)
        return class_id

    def convert_yolo_result_to_detection_result(self, yolo_output_results, timestamp, sensor_type):
        """
        Convert YOLO output results to DetectionResult2D.
        :param yolo_output_results: YOLO output results
        :param timestamp: timestamp of the image
        :param sensor_type: type of the sensor
        :return: DetectionResult2D instance
        """
        # Convert YOLO output to DetectionResult2D
        bboxes = yolo_output_results[0].boxes.xyxy.cpu().numpy()
        confs = yolo_output_results[0].boxes.conf.cpu().numpy()
        class_ids = yolo_output_results[0].boxes.cls.cpu().numpy()
        
        detected_objects = []
        for i in range(len(bboxes)):
            box = bboxes[i]
            class_id = self.map_class_id(int(class_ids[i]))
            score = float(confs[i])
            # calculate the center coordinates of the bounding box
            center_x = float((box[0] + box[2]) / 2)
            center_y = float((box[1] + box[3]) / 2)
            # print(class_id)
            detected_object = DetectedObject2D(
                box=[float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                class_id=class_id,
                score=score,
                pixel_bottom_center=[center_x, center_y],
            )
            detected_objects.append(detected_object)
        
        return detected_objects
    
    def detect(self, image: ndarray, timestamp, sensor_type, sensor_name) -> DetectionResult2D:
        image = self.apply_pre_mask(image, sensor_name)
        yolo_output_results = self.model(image, device=self.device, conf=self.confthre, iou=self.nmsthre, half=self.fp16, verbose=False, agnostic_nms=self.class_agnostic_nms)
        ## Convert results to DetectionResult2D
        detection_result = self.convert_yolo_result_to_detection_result(
            yolo_output_results,
            timestamp,
            sensor_type,
        )
        return self.apply_post_mask(detection_result, sensor_name, image.shape)

class Yolo26Detector(YoloDetector):
    """YOLOv2.6 detector for 2D images."""
    def __init__(self, model_path: Path, device: str = "cpu", confthre: float = 0.25, nmsthre: float = 0.45, fp16: bool = False, class_agnostic_nms: bool = False, pre_mask_path: Dict[str, Path] = None, post_mask_path: Dict[str, Path] = None, end2end: bool = False, id_mapping: Dict[int, int] = None):
        super().__init__(model_path, device, confthre, nmsthre, fp16, class_agnostic_nms, pre_mask_path, post_mask_path, id_mapping)

        self.end2end = end2end
    def detect(self, image: ndarray, timestamp, sensor_type, sensor_name) -> DetectionResult2D:
        image = self.apply_pre_mask(image, sensor_name)
        yolo_output_results = self.model(image, device=self.device, conf=self.confthre, iou=self.nmsthre, half=self.fp16, verbose=False, agnostic_nms=self.class_agnostic_nms, end2end=self.end2end)
        ## Convert results to DetectionResult2D
        detection_result = self.convert_yolo_result_to_detection_result(
            yolo_output_results,
            timestamp,
            sensor_type,
        )
        return self.apply_post_mask(detection_result, sensor_name, image.shape)

class Yolo26OBBDetector(Yolo26Detector):
    """YOLOv2.6 OBB detector for 2D images."""
    def __init__(self, model_path: Path, device: str = "cpu", confthre: float = 0.25, nmsthre: float = 0.45, fp16: bool = False, class_agnostic_nms: bool = False, pre_mask_path: Dict[str, Path] = None, post_mask_path: Dict[str, Path] = None, end2end: bool = False, id_mapping: Dict[int, int] = None):
        super().__init__(model_path, device, confthre, nmsthre, fp16, class_agnostic_nms, pre_mask_path, post_mask_path, end2end, id_mapping)

    def convert_yolo_result_to_detection_result(self, yolo_output_results, timestamp, sensor_type):
        """
        Convert YOLO output results to DetectionResult2D.
        :param yolo_output_results: YOLO output results
        :param timestamp: timestamp of the image
        :param sensor_type: type of the sensor
        :return: DetectionResult2D instance
        """
        # Convert YOLO output to DetectionResult2D
        bboxes = yolo_output_results[0].obb.xyxyxyxy.cpu().numpy()
        confs = yolo_output_results[0].obb.conf.cpu().numpy()
        class_ids = yolo_output_results[0].obb.cls.cpu().numpy()
        
        detected_objects = []
        for i in range(len(bboxes)):
            box = bboxes[i]
            class_id = self.map_class_id(int(class_ids[i]))
            score = float(confs[i])
            # calculate the center coordinates of the bounding box
            center = box.mean(axis=0)
            center_x = float(center[0])
            center_y = float(center[1])

            detected_object = DetectedObject2D(
                box=[float(box[0][0]), float(box[0][1]), float(box[1][0]), float(box[1][1]), float(box[2][0]), float(box[2][1]), float(box[3][0]), float(box[3][1])],
                class_id=class_id,
                score=score,
                pixel_bottom_center=[center_x, center_y],
            )
            detected_objects.append(detected_object)
        
        return detected_objects

class Yolo26OBBPedestrianDetector(Yolo26OBBDetector):
    """YOLOv2.6 OBB pedestrian detector for 2D images."""
    def __init__(self, model_path: Path, camera_center: Dict[str, List[int]], device: str = "cpu", confthre: float = 0.25, nmsthre: float = 0.45, fp16: bool = False, class_agnostic_nms: bool = False, pre_mask_path: Dict[str, Path] = None, post_mask_path: Dict[str, Path] = None, end2end: bool = False, id_mapping: Dict[int, int] = None):
        super().__init__(model_path, device, confthre, nmsthre, fp16, class_agnostic_nms, pre_mask_path, post_mask_path, end2end, id_mapping)

        self.camera_center = camera_center
    def convert_yolo_result_to_detection_result(self, yolo_output_results, timestamp, sensor_type, sensor_name):
        """
        Convert YOLO output results to DetectionResult2D.
        :param yolo_output_results: YOLO output results
        :param timestamp: timestamp of the image
        :param sensor_type: type of the sensor
        :param sensor_name: name of the sensor
        :return: DetectionResult2D instance
        """
        # Convert YOLO output to DetectionResult2D
        bboxes = yolo_output_results[0].obb.xyxyxyxy.cpu().numpy()
        confs = yolo_output_results[0].obb.conf.cpu().numpy()
        
        detected_objects = []
        for i in range(len(bboxes)):
            box = bboxes[i]
            score = float(confs[i])
            # calculate the bottom point of the pedestrian
            # center = self.predict_bottom_from_obb_box(box, tuple(self.camera_center[sensor_name]))
            # center_x = float(center[0])
            # center_y = float(center[1])
            center = box.mean(axis=0)
            center_x = float(center[0])
            center_y = float(center[1])

            detected_object = DetectedObject2D(
                box=[float(box[0][0]), float(box[0][1]), float(box[1][0]), float(box[1][1]), float(box[2][0]), float(box[2][1]), float(box[3][0]), float(box[3][1])],
                class_id=self.map_class_id(4),
                score=score,
                pixel_bottom_center=[center_x, center_y],
            )
            detected_objects.append(detected_object)
        
        return detected_objects
    
    def detect(self, image: ndarray, timestamp, sensor_type, sensor_name) -> DetectionResult2D:
        image = self.apply_pre_mask(image, sensor_name)
        yolo_output_results = self.model(image, device=self.device, conf=self.confthre, iou=self.nmsthre, half=self.fp16, verbose=False, agnostic_nms=self.class_agnostic_nms, end2end=self.end2end)
        ## Convert results to DetectionResult2D
        detection_result = self.convert_yolo_result_to_detection_result(
            yolo_output_results,
            timestamp,
            sensor_type,
            sensor_name
        )
        return self.apply_post_mask(detection_result, sensor_name, image.shape)
    
    def predict_bottom_from_obb_box(self, corners: ndarray, image_center: tuple[int, int]) -> List[float]:
        """Return the pedestrian bottom-center (x, y) given a pedestrian OBB.
        :param corners: The four OBB vertices, ordered around the rectangle.
        :param image_center: (cx, cy) of the source image.

        :return: The bottom-center (x, y) in image coordinates.
        """
        center = corners.mean(axis=0)
        direction = np.array([image_center[0] - center[0],
                              image_center[1] - center[1]], dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return [float(center[0]), float(center[1])]
        direction /= norm

        s_best = np.inf
        for i in range(4):
            a = corners[i].astype(np.float64)
            b = corners[(i + 1) % 4].astype(np.float64)
            s = self._ray_segment_intersect(center, direction, a, b)
            if s < s_best:
                s_best = s

        if not np.isfinite(s_best):
            return [float(center[0]), float(center[1])]

        hit = center + s_best * direction
        return [float(hit[0]), float(hit[1])]
    
    def _ray_segment_intersect(self, origin: np.ndarray, direction: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        e = b - a
        denom = direction[0] * e[1] - direction[1] * e[0]   
        if abs(denom) < 1e-12:
            return np.inf                                   
        diff = a - origin
        s = (diff[0] * e[1] - diff[1] * e[0]) / denom      
        t = (diff[0] * direction[1] - diff[1] * direction[0]) / denom
        if s < -1e-9 or t < -1e-9 or t > 1 + 1e-9:
            return np.inf
        return max(s, 0.0)