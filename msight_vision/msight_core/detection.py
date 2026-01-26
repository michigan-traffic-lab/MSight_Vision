from msight_core.nodes import DataProcessingNode, NodeConfig
from msight_core.data import ImageData, DetectionResultsData
import yaml
from pathlib import Path
import numpy as np
from .. import YoloDetector, HashLocalizer, ClassicWarperWithExternalUpdate
import torch
import time
from msight_core.utils import get_redis_client


device = "cuda" if torch.cuda.is_available() else "cpu"

def load_locmaps(loc_maps_path):
    """
    Load localization maps from the specified path.
    :param loc_maps_path: path to the localization maps in the config file
    :return: localization maps
    """
    result = {key: np.load(item) for key, item in loc_maps_path.items()}
    return result

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
        self.model_path = self.det_config["model_config"]["ckpt_path"]
        self.confthre = self.det_config["model_config"]["confthre"]
        self.nmsthre = self.det_config["model_config"]["nmsthre"]
        self.class_agnostic_nms = self.det_config["model_config"]["class_agnostic_nms"]
        self.logger.info(f"Initializing YoloOneStageDetectionNode with model path: {self.model_path}, no_warp: {self.no_warp}, warper_matrix_redis_prefix: {self.warper_matrix_redis_prefix}, confthre: {self.confthre}, nmsthre: {self.nmsthre}, class_agnostic_nms: {self.class_agnostic_nms}")
        self.detector = YoloDetector(model_path=Path(self.model_path), device=device, confthre=self.confthre, nmsthre=self.nmsthre, fp16=False, class_agnostic_nms=self.class_agnostic_nms)
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
        result = self.detector.detect(image, timestamp, self.sensor_type)
        localizer = self.localizers[sensor_name]
        localizer.localize(result)
        self.logger.info(f"Detection completed in {time.time() - start:.2f} seconds for sensor: {sensor_name}")
        def is_number(val):
            return isinstance(val, (int, float, np.number)) and np.isfinite(val)
        result.object_list = [obj for obj in result.object_list if is_number(obj.lat) and is_number(obj.lon)]
        raw_sensor_data = None
        if self.include_sensor_data_in_result:
            raw_sensor_data = data
        detection_result_data = DetectionResultsData(result, image_frame_id=data.frame_id, capture_timestamp=data.capture_timestamp, creation_timestamp=time.time(), sensor_name=sensor_name, raw_sensor_data=raw_sensor_data)
        # print(f"Detection results: {detection_result_data.to_dict()}")
        return detection_result_data
