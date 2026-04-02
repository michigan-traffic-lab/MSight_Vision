## python main.py -c config.yaml
from msight_vision.utils import ImageRetriever
from msight_vision import MergedDetector, HashLocalizer, SortTracker, ClassicWarper
from msight_vision.fuser import StateEllsworthFuser
from msight_vision.state_estimator import FiniteDifferenceStateEstimator
from msight_base import Frame
import argparse
from pathlib import Path
import cv2
import torch
from utils import plot_2d_detection_results, load_locmaps, is_number
import yaml
from msight_base.visualizer import Visualizer

argparser = argparse.ArgumentParser(description="roundabout perception example")
argparser.add_argument("-c", "--config", type=Path, required=True, help="config file path")
args = argparser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

### image directory###
img_dir = Path(config["data_path"])
img_retriever = ImageRetriever(img_dir=img_dir)

### device
device = "cuda" if torch.cuda.is_available() else "cpu"

### initialize warper
no_warp = config['warper_config']['no_warp']
if not no_warp:
    std_imgs = config['warper_config']['std_imgs']
    warpers= {}
    for sensor_name, img_path in std_imgs.items():
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error: {img_path} is not a valid image file.")
            exit(1)
        warpers[sensor_name] = ClassicWarper(img)

### initialize detector
detector = MergedDetector(model_config=config['model_config'], device=device)

### initialize localizer
loc_maps_path = config ["loc_maps"]
loc_maps = load_locmaps(loc_maps_path)
localizers = {key: HashLocalizer(item['x_map'], item['y_map']) for key, item in loc_maps.items()}

### initialize fuser
fuser = StateEllsworthFuser()

### initialize tracker
tracker = SortTracker()

### initialize state estimator
state_estimator = FiniteDifferenceStateEstimator()

### initialize visualizer
visualizer = Visualizer("./basemap_configs/roundabout.jpg")

step = 0
while True:
    img_buff = img_retriever.get_image()
    if img_buff is None:
        break
    
    ## detection
    detection_buffer = {}
    for sensor_name in img_buff.keys():
        if not no_warp:
            warper = warpers[sensor_name]
            img = warper.warp(img_buff[sensor_name]["image"])
        else:
            img = img_buff[sensor_name]["image"]
        result = detector.detect(img, img_buff[sensor_name]["timestamp"], "fisheye")
        detection_buffer[sensor_name] = result
    #print(f"Detection result: {detection_buffer}")
    
    ## localization
    for sensor_name, detection_result in detection_buffer.items():
        localizer = localizers[sensor_name]
        localizer.localize(detection_result)

    ## remove those objects that are not localized (obj.lat isn't a number like None, inf, -inf)
    for sensor_name, detection_result in detection_buffer.items():
        detection_result.object_list = [obj for obj in detection_result.object_list if is_number(obj.lat) and is_number(obj.lon)]

    ## fusion
    fusion_result = fuser.fuse(detection_buffer)
    # print(fusion_result)

    ## tracking
    tracking_result = tracker.track(fusion_result)

    ## state estimation
    result = state_estimator.estimate(tracking_result)

    ## visualization
    # creating frame
    result_frame = Frame(step)
    for obj in result:
        result_frame.add_object(obj)
    vis_img = visualizer.render(result_frame , with_traj=True)
    cv2.imshow("Visualization", vis_img)
    key = cv2.waitKey(100)
    # detection2d_results_img = plot_2d_detection_results(img_buff, detection_buffer, grid_size=(2, 2), size=(1280, 960))
    # cv2.imshow("Detection Results", detection2d_results_img)
    # key = cv2.waitKey(1)

    step += 1

cv2.destroyAllWindows() 
