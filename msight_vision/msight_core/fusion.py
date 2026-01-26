from msight_core.nodes import DataProcessingNode, NodeConfig
from msight_core.data import RoadUserListData, DetectionResultsData
from pathlib import Path
import yaml
import importlib.util
import sys
import copy

def load_class_from_file(file_path: str, class_path: str):
    module_name = Path(file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Resolve class from class_path
    components = class_path.split(".")
    cls = module
    for comp in components[1:]:  # Skip the module name
        cls = getattr(cls, comp)
    return cls

class FuserNode(DataProcessingNode):
    default_configs = NodeConfig(
        publish_topic_data_type=RoadUserListData
    )
    def __init__(self, configs, fusion_configs_path):
        super().__init__(configs)
        self.config_file_path = Path(fusion_configs_path)
        with open(self.config_file_path, "r") as f:
            self.fusion_configs = yaml.safe_load(f)
        self.fuser_class_path = self.fusion_configs["fuser_config"]["class_path"]
        self.fuser_file_path = self.fusion_configs["fuser_config"]["file_path"]
        self.sensor_list = self.fusion_configs["fuser_config"]["sensor_list"]
        FuserClass = load_class_from_file(self.fuser_file_path, self.fuser_class_path)
        self.fuser = FuserClass()
        self.buffer = {sensor: None for sensor in self.sensor_list}
        assert configs.sensor_name is not None, "sensor_name must be provided in configs for the fusion node."
        self.sensor_name = self.configs.sensor_name

    def process(self, data: DetectionResultsData):
        self.buffer[data.sensor_name] = data.detection_result
        # print(data.detection_result)
        buffer = copy.copy(self.buffer)
        self.logger.info(f"Processing detection results from sensor: {data.sensor_name}")
        sensor_name = data.sensor_name
        if sensor_name not in self.sensor_list:
            raise ValueError(f"Sensor {sensor_name} not in configured sensor list: {self.sensor_list}")

        for _, detection_result in buffer.items():
            if detection_result is None:
                return None
        self.logger.info(f"All data received. Fusing detection results for sensor.")
        fused_result = self.fuser.fuse(buffer)
        road_user_list_data = RoadUserListData(
            road_user_list=fused_result,
            capture_timestamp=data.capture_timestamp,
            sensor_name=self.sensor_name
        )
        self.buffer = {sensor: None for sensor in self.sensor_list}  # Reset buffer after fusion
        # print(road_user_list_data)
        return road_user_list_data
