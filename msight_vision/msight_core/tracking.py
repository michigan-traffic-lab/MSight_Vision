from msight_core.nodes import DataProcessingNode, NodeConfig
from msight_core.data import RoadUserListData
from .. import SortTracker
from pathlib import Path
import yaml
import time


class SortTrackerNode(DataProcessingNode):
    default_configs = NodeConfig(
        publish_topic_data_type=RoadUserListData
    )
    def __init__(self, configs, tracking_configs_path):
        super().__init__(configs)
        self.config_file_path = Path(tracking_configs_path)
        with open(self.config_file_path, "r") as f:
            self.tracking_configs = yaml.safe_load(f)
        self.tracker = SortTracker()
        

    def process(self, data: RoadUserListData) -> RoadUserListData:
        self.logger.info(f"Processing road user list data from sensor: {data.sensor_name}")
        start = time.time()
        road_user_list = data.road_user_list
        # print(road_user_list)
        tracking_result = self.tracker.track(road_user_list)
        data.road_user_list = tracking_result
        self.logger.info(f"Tracking completed in {time.time() - start:.2f} seconds for sensor: {data.sensor_name}")
        return data

    