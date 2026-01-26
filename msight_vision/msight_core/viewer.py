from msight_core.nodes import SinkNode
from msight_core.data import RoadUserListData, DetectionResultsData
from msight_base.visualizer import Visualizer
# from pathlib import Path
# import yaml
from msight_base import Frame
import cv2


class RoadUserListViewerNode(SinkNode):
    def __init__(self, configs, basemap_path, with_traj=True, show_heading=False):
        super().__init__(configs)
        self.basemap_path = basemap_path
        self.visualizer = Visualizer(basemap_path)
        self.with_traj = with_traj
        self.show_heading = show_heading
        self.step=0


    def on_message(self, data: RoadUserListData):
        self.logger.info(f"Received road user list data from sensor: {data.sensor_name}")
        road_user_list = data.road_user_list
        if not self.with_traj:
            # print(f"fused result: {len(fused_result)} objects")
            vis_image = self.visualizer.render_roaduser_points(road_user_list)
            cv2.imshow(self.name, vis_image)
            cv2.waitKey(1)
            return
        result_frame = Frame(self.step)  
        for obj in road_user_list:
            # print(obj.traj_id)
            result_frame.add_object(obj)
        vis_img = self.visualizer.render(result_frame, with_traj=self.with_traj, show_heading=self.show_heading)
        cv2.imshow(self.name, vis_img)
        cv2.waitKey(1)
        self.step += 1

class DetectionResults2DViewerNode(SinkNode):
    def __init__(self, configs):
        super().__init__(configs)

    def on_message(self, data: DetectionResultsData):
        # print(data)
        self.logger.info(f"Received detection results from sensor: {data.sensor_name}")
        raw_image_data = data.raw_sensor_data
        decoded_image = raw_image_data.to_ndarray()
        detection_result = data.detection_result
        for obj in detection_result.object_list:
            # print(obj.box, obj.pixel_bottom_center)
            x1, y1, x2, y2 = map(int, obj.box)
            px, py = map(int, obj.pixel_bottom_center)
            cv2.circle(decoded_image, (px, py), 5, (0, 0, 255), -1)
            cv2.rectangle(decoded_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow(self.name, decoded_image)
        cv2.waitKey(1)