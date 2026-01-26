from msight_base import RoadUserPoint


def detection_to_roaduser_point(detected_object, sensor_id):
    """
    Convert detection result to RoadUserPoint.
    :param detection_result: DetectionResult2D instance
    :return: list of RoadUserPoint instances
    """
    # print(detected_object.lat, detected_object.lon, detected_object.class_id, detected_object.score)
    road_user_point = RoadUserPoint(
        x = detected_object.lat,
        y = detected_object.lon,
        category=detected_object.class_id,
        confidence=detected_object.score,
    )
    road_user_point.sensor_data[sensor_id] = detected_object.to_dict()
    return road_user_point
