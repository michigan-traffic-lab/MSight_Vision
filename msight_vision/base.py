from msight_base import DetectionResultBase, DetectedObjectBase, RoadUserPoint
import numpy as np
from typing import List, Dict

class DetectedObject2D(DetectedObjectBase):
    """Detected object for 2D images."""

    def __init__(self, box: list, class_id: int, score: float, pixel_bottom_center: List[float], obj_id: str = None, lat: float = None, lon: float = None, x: float = None, y: float = None):
        """
        Initialize the detected object.
        :param box: bounding box coordinates (x1, y1, x2, y2)
        :param class_id: class ID of the detected object
        :param score: confidence score of the detection
        :param obj_id: unique ID of the detected object (optional)
        :param lat: latitude of the detected object (optional)
        :param lon: longitude of the detected object (optional)
        :param x: x coordinate in the coordination of interest like utm of the detected object (optional)
        :param y: y coordinate in the coordination of intererst like utm of the detected object (optional)
        """
        super().__init__()
        self.box = box
        self.class_id = class_id
        self.score = score
        self.pixel_bottom_center = pixel_bottom_center
        self.obj_id = obj_id
        self.lat = lat
        self.lon = lon
        self.x = x
        self.y = y

    def to_dict(self):
        """
        Convert the detected object to a dictionary.
        :return: dictionary representation of the detected object
        """
        return {
            "box": self.box,
            "class_id": self.class_id,
            "score": self.score,
            "obj_id": self.obj_id,
            "lat": self.lat,
            "lon": self.lon,
            "x": self.x,
            "y": self.y,
            "pixel_bottom_center": self.pixel_bottom_center,
        }
    
    @staticmethod
    def from_dict(data: dict):
        """
        Create a DetectedObject2D instance from a dictionary.
        :param data: dictionary representation of the detected object
        :return: DetectedObject2D instance
        """
        return DetectedObject2D(
            box=data["box"],
            class_id=data["class_id"],
            score=data["score"],
            obj_id=data.get("obj_id"),
            lat=data.get("lat") or None,
            lon=data.get("lon") or None,
            x=data.get("x") or None,
            y=data.get("y") or None,
            pixel_bottom_center=data.get("pixel_bottom_center") or None,
        )

    def __repr__(self):
        return f"DetectedObject2D(box={self.box}, class_id={self.class_id}, score={self.score}, obj_id={self.obj_id}, lat={self.lat}, lon={self.lon}, x={self.x}, y={self.y})" 

class DetectionResult2D(DetectionResultBase):
    """Detection result for 2D images."""

    def __init__(self, object_list: List[DetectedObject2D], timestamp: int, sensor_type: str):
        """
        Initialize the detection result.
        :param detected_objects: list of detected objects
        """
        super().__init__(object_list, timestamp, sensor_type)

class ImageDetector2DBase:
    def detect(self, image: np.ndarray) -> DetectionResult2D:
        """
        Detector base, a detector detects objects in the image.
        :param image: input image
        :return: list of detected objects
        """
        raise NotImplementedError("detect method not implemented")
    
class TrackerBase:
    def __init__(self):
        pass

    def track(self, list) ->Dict[str, RoadUserPoint]:
        """
        Track the detected objects in the image.
        :param detection_result: DetectionResult2D instance
        :return: updated DetectionResult2D instance with tracking information
        """
        raise NotImplementedError("track method not implemented")
