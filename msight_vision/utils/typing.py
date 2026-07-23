from msight_base import RoadUserPoint


def detection_to_roaduser_point(detected_object, sensor_id):
    """
    Convert detection result to RoadUserPoint.
    :param detection_result: DetectionResult2D instance
    :return: list of RoadUserPoint instances
    """
    road_user_point = RoadUserPoint(
        x = detected_object.lat,
        y = detected_object.lon,
        category=detected_object.class_id,
        confidence=detected_object.score,
    )
    # ASSIGN a fresh dict instead of mutating RoadUserPoint's shared mutable
    # default argument (sensor_data={}): the old in-place write made every point
    # alias one process-global dict — unbounded growth and every point carrying
    # all prior detections' data
    road_user_point.sensor_data = {sensor_id: detected_object.to_dict()}
    return road_user_point
