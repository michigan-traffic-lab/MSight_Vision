from msight_core.nodes import DataProcessingNode, NodeConfig
from msight_core.data import RoadUserListData
from .. import FiniteDifferenceStateEstimator
from pathlib import Path
import yaml
import time

class  FiniteDifferenceStateEstimatorNode(DataProcessingNode):
    default_configs = NodeConfig(
        publish_topic_data_type=RoadUserListData
    )
    def __init__(self, configs, state_estimation_configs_path):
        super().__init__(configs)
        self.config_file_path = Path(state_estimation_configs_path)
        with open(self.config_file_path, "r") as f:
            self.state_estimation_configs = yaml.safe_load(f)
        self.frame_rate = self.state_estimation_configs["state_estimator_config"].get("frame_rate", 5)
        self.frame_interval = self.state_estimation_configs["state_estimator_config"].get("frame_interval", 1)
        self.dist_threshold = self.state_estimation_configs["state_estimator_config"].get("dist_threshold", 4)
        self.state_estimator = FiniteDifferenceStateEstimator(
            frame_rate=self.frame_rate,
            frame_interval=self.frame_interval,
            dist_threshold=self.dist_threshold
        )

    def process(self, data: RoadUserListData) -> RoadUserListData:
        self.logger.info(f"Processing road user list data from sensor: {data.sensor_name}")
        start = time.time()
        road_user_list = data.road_user_list
        result = self.state_estimator.estimate(road_user_list)
        # road_user_list_data = RoadUserListData(
        #     road_user_list=result,
        #     capture_timestamp=data.capture_timestamp,
        #     sensor_name=data.sensor_name
        # )
        data.road_user_list = result
        self.logger.info(f"State Estimation completed in {time.time() - start:.2f} seconds for sensor: {data.sensor_name}")
        return data