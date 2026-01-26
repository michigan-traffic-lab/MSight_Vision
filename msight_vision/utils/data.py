from pathlib import Path
from datetime import datetime
import cv2

def get_time_from_name(name: str) -> datetime:
    """
    Get time from name.
    :param name: name of the file
    :return: time in datetime format
    """
    # name format is 2023-12-19 11-00-00-013294.jpg
    time_str = name.split(".")[0].split("#")[0]
    time_str = time_str.replace("-", " ").replace(" ", "-")
    return datetime.strptime(time_str, "%Y-%m-%d-%H-%M-%S-%f")

class ImageRetriever:
    def __init__(self, img_dir: Path, sensor_list:list = None, time_tolerance: float = 0.2):
        self.img_dir = img_dir
        self.img_buff = {}
        self.timestamps = {}
        self.time_tolerance = time_tolerance
        if sensor_list is None:
            # find all folders in the directory
            self.sensor_list = [f.name for f in img_dir.iterdir() if f.is_dir()]
        else:
            self.sensor_list = sensor_list
        for sensor_name in self.sensor_list:
            # find all folders in the directory
            self.img_buff[sensor_name] = [f for f in (img_dir / sensor_name).iterdir() if f.is_file()]
            # sort the image files by name
            self.img_buff[sensor_name].sort(key=lambda x: x.name.split("#")[0])
            self.timestamps[sensor_name] = [get_time_from_name(f.name).timestamp() for f in self.img_buff[sensor_name]]

        self.length = min([len(self.img_buff[sensor_name]) for sensor_name in self.sensor_list])
        self.step = 0
        self.main_sensor = self.sensor_list[0]

    def _find_closest_timestamp(self, timestamps, target):
        """Binary search to find the index of the closest timestamp to the target."""
        low, high = 0, len(timestamps) - 1
        best_idx = -1
        best_diff = float('inf')
        while low <= high:
            mid = (low + high) // 2
            diff = abs(timestamps[mid] - target)
            if diff < best_diff:
                best_diff = diff
                best_idx = mid
            if timestamps[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return best_idx, timestamps[best_idx] if best_idx != -1 else None, best_diff

    def get_image(self):
        print(f"Retrieving images at step {self.step}/{self.length}")
        if self.step >= self.length:
            return None
        result = {}
        current_time = self.timestamps[self.main_sensor][self.step]
        for sensor_name in self.sensor_list:
            # find the index of the closest timestamp in the sensor's timestamp list, use binary search
            idx, closest_timestamp, best_diff = self._find_closest_timestamp(self.timestamps[sensor_name], current_time)
            if best_diff > self.time_tolerance:
                print(f"Warning: No close timestamp found for sensor {sensor_name} at time {current_time}. Closest time is {closest_timestamp} with difference {best_diff}.")
            img_path = self.img_buff[sensor_name][idx]
            # read image as numpy array
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Error: {img_path} is not a valid image file.")
                return None
            result[sensor_name] = {}
            result[sensor_name]["image"] = img
            ## get time from name
            ## datetime needs to be converted to timestamp
            result[sensor_name]["timestamp"] = get_time_from_name(img_path.name).timestamp()
            result[sensor_name]["path"] = img_path
            result[sensor_name]["frame_id"] = img_path.stem.split("#")[-1] if "#" in img_path.stem else None
        self.step += 1
        return result
