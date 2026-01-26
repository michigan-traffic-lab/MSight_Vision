from typing import Dict, List
from .base import DetectionResult2D
# from geopy.distance import geodesic
# from msight_base import RoadUserPoint
from .utils import detection_to_roaduser_point
from msight_base import RoadUserPoint

class FuserBase:
    def fuse(self, results: Dict[str, DetectionResult2D]) -> List[RoadUserPoint]:
        """
        Fuses the data from different sources into a single output.
        :param data: The input data to be fused.
        :return: The fused output.
        """
        raise NotImplementedError("FuserBase is an abstract class and cannot be instantiated directly.")
    
## This is a simple example of a fuser that combines the results from different cameras, which works at the roundabout of State and Ellsworth in Smart Intersection Project.
class StateEllsworthFuser:
    '''
    This is a simple example of a fuser that combines the results from different cameras, which works at the roundabouot of State and Ellsworth.
    '''
    def __init__(self):
        self.lat1 = 42.229379
        self.lon1 = -83.739003
        self.lat2 = 42.229444
        self.lon2 = -83.739013

    def fuse(self, detection_buffer: Dict[str, DetectionResult2D]) -> List[RoadUserPoint]:
        fused_vehicle_list = []

        vehicle_list = detection_buffer['gs_State_Ellsworth_NW'].object_list
        for v in vehicle_list: # cam_ne
            if v.lat > self.lat1 and v.lon > self.lon1:
                fused_vehicle_list.append(detection_to_roaduser_point(v, 'gs_State_Ellsworth_NW'))

        vehicle_list = detection_buffer['gs_State_Ellsworth_NE'].object_list
        for v in vehicle_list: # cam_nw
            if v.lat > self.lat2 and v.lon < self.lon2:
                fused_vehicle_list.append(detection_to_roaduser_point(v, 'gs_State_Ellsworth_NE'))

        vehicle_list = detection_buffer['gs_State_Ellsworth_SE'].object_list
        for v in vehicle_list: # cam_se
            if v.lat < self.lat1 and v.lon > self.lon1:
                fused_vehicle_list.append(detection_to_roaduser_point(v, 'gs_State_Ellsworth_SE'))

        vehicle_list = detection_buffer['gs_State_Ellsworth_SW'].object_list
        for v in vehicle_list: # cam_sw
            if v.lat < self.lat2 and v.lon < self.lon2:
                fused_vehicle_list.append(detection_to_roaduser_point(v, 'gs_State_Ellsworth_SW'))
        return fused_vehicle_list

