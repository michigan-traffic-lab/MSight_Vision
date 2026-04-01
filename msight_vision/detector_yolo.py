from numpy import ndarray
from msight_vision.base import DetectionResult2D, DetectedObject2D
from .base import ImageDetector2DBase
from ultralytics import YOLO
from pathlib import Path

class YoloDetector(ImageDetector2DBase):
    """YOLOv5 detector for 2D images."""

    def __init__(self, model_path: Path, device: str = "cpu", confthre: float = 0.25, nmsthre: float = 0.45, fp16: bool = False, class_agnostic_nms: bool = False):
        """
        Initialize the YOLO detector.
        :param model_path: path to the YOLO model
        :param device: device to run the model on (e.g., 'cpu', 'cuda')
        """
        super().__init__()
        self.model = YOLO(str(model_path))
        self.device = device
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.fp16 = fp16
        self.class_agnostic_nms = class_agnostic_nms
        

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
            class_id = int(class_ids[i])
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
        
        detection_result = DetectionResult2D(
            detected_objects,
            timestamp,
            sensor_type,
        )
        
        return detection_result
    
    def detect(self, image: ndarray, timestamp, sensor_type) -> DetectionResult2D:
        yolo_output_results = self.model(image, device=self.device, conf=self.confthre, iou=self.nmsthre, half=self.fp16, verbose=False, agnostic_nms=self.class_agnostic_nms)
        ## Convert results to DetectionResult2D
        detection_result = self.convert_yolo_result_to_detection_result(
            yolo_output_results,
            timestamp,
            sensor_type,
        )
        return detection_result
    
class Yolo26Detector(YoloDetector):
    """YOLOv2.6 detector for 2D images."""
    def __init__(self, model_path: Path, device: str = "cpu", confthre: float = 0.25, nmsthre: float = 0.45, fp16: bool = False, class_agnostic_nms: bool = False, end2end: bool = False):
        super().__init__(model_path, device, confthre, nmsthre, fp16, class_agnostic_nms)

        self.end2end = end2end
    def detect(self, image: ndarray, timestamp, sensor_type) -> DetectionResult2D:
        yolo_output_results = self.model(image, device=self.device, conf=self.confthre, iou=self.nmsthre, half=self.fp16, verbose=False, agnostic_nms=self.class_agnostic_nms, end2end=self.end2end)
        ## Convert results to DetectionResult2D
        detection_result = self.convert_yolo_result_to_detection_result(
            yolo_output_results,
            timestamp,
            sensor_type,
        )
        return detection_result

class Yolo26OBBDetector(Yolo26Detector):
    """YOLOv2.6 OBB detector for 2D images."""
    def __init__(self, model_path: Path, device: str = "cpu", confthre: float = 0.25, nmsthre: float = 0.45, fp16: bool = False, class_agnostic_nms: bool = False, end2end: bool = False):
        super().__init__(model_path, device, confthre, nmsthre, fp16, class_agnostic_nms, end2end)

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
            class_id = int(class_ids[i])
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
        
        detection_result = DetectionResult2D(
            detected_objects,
            timestamp,
            sensor_type,
        )
        
        return detection_result
    
    def detect(self, image: ndarray, timestamp, sensor_type) -> DetectionResult2D:
        yolo_output_results = self.model(image, device=self.device, conf=self.confthre, iou=self.nmsthre, half=self.fp16, verbose=False, agnostic_nms=self.class_agnostic_nms, end2end=self.end2end)
        ## Convert results to DetectionResult2D
        detection_result = self.convert_yolo_result_to_detection_result(
            yolo_output_results,
            timestamp,
            sensor_type,
        )
        return detection_result
