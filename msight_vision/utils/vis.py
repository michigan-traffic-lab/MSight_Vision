import cv2
def visualize_detection_result(image, detection_result, box_color=(0, 255, 0), text_color=(0, 255, 0)):
    """
    Visualize the detection result on the image.
    :param image: input image as a numpy array
    :param detection_result: DetectionResult2D instance
    :return: image with bounding boxes drawn
    """
    for obj in detection_result.object_list:
        box = obj.box
        class_id = obj.class_id
        score = obj.score
        # draw rectangle
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), box_color, 2)
        # put class_id and score
        cv2.putText(image, f"{class_id}:{score:.2f}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
    return image
