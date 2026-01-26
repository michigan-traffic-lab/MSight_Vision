from numpy import ndarray
from msight_det2d.base import DetectionResult2D, DetectedObject2D
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
