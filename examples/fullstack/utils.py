import cv2
import numpy as np

def build_image_grid(images, grid_size, size=(1280, 960)):
    """
    Build a grid of images.
    :param images: list of images
    :param grid_size: size of the grid (rows, cols)
    :return: grid image
    """
    rows, cols = grid_size
    # print(f"Building image grid with size: {grid_size}")
    img_height, img_width = images[0].shape[:2]
    grid_img = cv2.vconcat([cv2.hconcat(images[i * cols:(i + 1) * cols]) for i in range(rows)])
    # Resize the grid image to the specified size
    grid_img = cv2.resize(grid_img, size)
    return grid_img

def plot_2d_detection_results(img_buffer, detection_buffer, grid_size=(2,2), size=(1280, 960)):
    """
    Plot 2D detection results on images.
    :param img_buffer: image buffer
    :param detection_buffer: detection buffer
    :param grid_size: size of the grid (rows, cols)
    :return: None
    """
    images = []
    for sensor_name in img_buffer.keys():
        img = img_buffer[sensor_name]["image"]
        timestamp = img_buffer[sensor_name]["timestamp"]
        # Draw bounding boxes on the image
        for detected_object in detection_buffer[sensor_name].object_list:
            box = detected_object.box
            class_id = detected_object.class_id
            score = detected_object.score
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(img, f"ID: {class_id}, Score: {score:.2f}", (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        images.append(img)

    # Build a grid of images and display it
    grid_img = build_image_grid(images, grid_size, size=size)
    return grid_img

def load_locmaps(loc_maps_path):
    """
    Load localization maps from the specified path.
    :param loc_maps_path: path to the localization maps in the config file
    :return: localization maps
    """
    result = {key: np.load(item) for key, item in loc_maps_path.items()}
    return result

def is_number(val):
    return isinstance(val, (int, float, np.number)) and np.isfinite(val)
