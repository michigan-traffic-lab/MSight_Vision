from typing import List
from msight_base import RoadUserPoint, TrajectoryManager
from geopy.distance import geodesic
import numpy as np


class StateEstimatorBase:
    def estimate(self, road_user_point_list: List[RoadUserPoint]) -> List[RoadUserPoint]:
        """
        Estimate the state of road users based on the provided list of RoadUserPoint instances.
        :param road_user_point_list: List of RoadUserPoint instances to estimate the state from.
        :return: Estimated state of road users.
        """
        raise NotImplementedError("StateEstimatorBase is an abstract class and cannot be instantiated directly.")
    
class FiniteDifferenceStateEstimator(StateEstimatorBase):
    def __init__(self, frame_rate=5, frame_interval=1, dist_threshold=2):
        """
        Initialize the finite difference state estimator.
        :param frame_rate: Frame rate of the video stream.
        :param frame_interval: Interval the two object to calculate the difference, the two neighbor objects has interval 0.
        :param heading_mask_area: Area to mask the heading estimation (optional).
        """
        self.frame_rate = frame_rate
        self.frame_interval = frame_interval
        self.trajectory_manager = TrajectoryManager(max_frames=100)
        self.dist_threshold = dist_threshold

    # def calc_heading(self, obj: RoadUserPoint, anchor: RoadUserPoint, scale):
        
    def get_anchor_point(self, obj):
        traj = obj.traj
        if len(traj.steps) <= 1:
            None
        if len(traj.steps) < self.frame_interval + 2:
            anchor_index = 0
        else:
            anchor_index = len(traj.steps) - self.frame_interval - 1
        anchor_step = traj.steps[anchor_index]
        anchor = traj.step_to_object_map[anchor_step]
        return anchor
    
    def calc_xy_difference(self, obj: RoadUserPoint, anchor: RoadUserPoint, scale="latlon"):
        """
        Calculate the difference between the object and the anchor point in the specified scale, sign is persistent.
        :param
        obj: RoadUserPoint instance representing the object.
        :param anchor: RoadUserPoint instance representing the anchor point.
        :param scale: Scale of the coordinates, either "latlon", "utm" or "meters". ("meters" and "utm" are equivalent)
        :return: Difference between the object and the anchor point in meters.
        """
        if scale == "latlon":
            lat1, lon1 = obj.x, obj.y
            lat2, lon2 = anchor.x, anchor.y
            # Calculate the distance in meters using geodesic
            dx = geodesic((lat1, lon1), (lat2, lon1)).meters
            if lat1 < lat2:
                dx = -dx
            dy = geodesic((lat1, lon1), (lat1, lon2)).meters
            if lon1 < lon2:
                dy = -dy    
        elif scale in ["utm", "meters"]:
            # just take difference
            dx = obj.x - anchor.x
            dy = obj.y - anchor.y
        else:
            raise ValueError("Invalid scale. Use 'latlon', 'utm' or 'meters'.")
        return dx, dy
    
    def calc_heading(self, dx: float, dy: float, temporal_distance, fallback) -> float:
        """
        Calculate the heading of the object based on the difference in x and y coordinates.
        :param
        dx: Difference in x coordinates.
        :param dy: Difference in y coordinates.
        :param temporal_distance: Temporal distance between the two points.
        :return: Heading of the object in degree.
        """
        # clock-wise, in degree
        # north: 0, east: 90, north-east: 45
        if temporal_distance > self.dist_threshold:
            return fallback
        heading = np.arctan2(dy, dx) * 180 / np.pi
        return heading
    
    def calc_speed(self, obj: RoadUserPoint, anchor: RoadUserPoint, temporal_distance) -> float:
        """
        Calculate the speed of the object based on the difference in x and y coordinates and temporal distance.
        :param
        obj: RoadUserPoint instance representing the object.
        :param anchor: RoadUserPoint instance representing the anchor point.
        :param temporal_distance: Temporal distance between the two points.
        :return: Speed of the object in meters per second.
        """
        
        step_now = obj.frame_step
        step_anchor = anchor.frame_step
        time_difference = (step_now - step_anchor) * 1 / self.frame_rate  # in seconds
        speed = (temporal_distance / time_difference) if time_difference > 0 else 0.0
        return speed



    def estimate(self, road_user_point_list: List[RoadUserPoint], scale="latlon") -> List[RoadUserPoint]:
        """
        Estimate the state of road users based on the provided list of RoadUserPoint instances.
        :param road_user_point_list: List of RoadUserPoint instances to estimate the state from.
        :param scale: Scale of the coordinates, either "latlon", "utm" or "meters". ("meters" and "utm" are equivalent)
        :return: Estimated state of road users.
        """
        self.trajectory_manager.add_list_as_new_frame(road_user_point_list)
        for obj in road_user_point_list:
            anchor = self.get_anchor_point(obj)
            if anchor is None:
                continue
            dx, dy = self.calc_xy_difference(obj, anchor, scale)
            temporal_distance = (dx**2 + dy**2)**0.5
            obj.heading = self.calc_heading(dx, dy, temporal_distance, anchor.heading) 
            obj.speed = self.calc_speed(obj, anchor, temporal_distance)

        return road_user_point_list
