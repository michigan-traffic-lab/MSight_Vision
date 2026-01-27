from msight_core.nodes import SinkNode, NodeConfig
from msight_core.data import ImageData
import cv2
import numpy as np
import time
from msight_core.utils import get_redis_client
from pathlib import Path
import yaml
import threading

class WarperMatrixUpdaterNode(SinkNode):
    def __init__(self, configs, warper_configs_path):
        '''
        The constructor for the WarperMatrixUpdaterNode class.
        :param standard_image_path: Path to the standard image used for warping.
        :param update_interval: Interval in steps between the updates.
        :param time_threshold: Time threshold in seconds to trigger an update.
        :param redis_prefix: Redis prefix for storing the warp matrix.
        '''
        super().__init__(configs)
        self.warper_config_path = Path(warper_configs_path)
        self.detector = None
        with open(self.warper_config_path, "r") as f:
            self.warper_config = yaml.safe_load(f)
        self.update_interval = self.warper_config["warper_config"]["update_interval"]
        self.time_threshold = self.warper_config["warper_config"]["time_threshold"]
        self.redis_prefix = self.warper_config["warper_config"]["redis_prefix"]
        standard_images_paths = self.warper_config["warper_config"]["std_imgs"]
        self.standard_images = {sensor_name: cv2.imread(img_path) for (sensor_name, img_path) in standard_images_paths.items()}
        self.steps = {sensor_name: 0 for sensor_name in standard_images_paths}
        self.last_update_times = {sensor_name: time.time() for sensor_name in standard_images_paths}

    def get_warp_matrix_between_two_image(self, im1, im2):

        # Convert images to grayscale
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        # Find size of image1
        # sz = im1.shape

        # Define the motion model
        warp_mode = cv2.MOTION_HOMOGRAPHY

        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Specify the number of iterations.
        number_of_iterations = 500

        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = 1e-10

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    number_of_iterations,  termination_eps)

        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC(
            im1_gray, im2_gray, warp_matrix, warp_mode, criteria)
        return warp_matrix
    
    def update_warp_matrix(self, image, sensor_name):
        standard_img = self.standard_images[sensor_name]
        warp_matrix = self.get_warp_matrix_between_two_image(
            standard_img, image)
        return warp_matrix
    
    def update_warp_matrix_in_redis(self, warp_matrix, sensor_name):
        redis_client = get_redis_client()
        # Convert the warp matrix to a string and store it in Redis
        warp_matrix_str = np.array2string(warp_matrix, separator=',')
        redis_client.set(self.redis_prefix + f":{sensor_name}", warp_matrix_str)
        # self.logger.info("Warp matrix updated in Redis.")

    def on_message(self, data: ImageData):
        sensor_name = data.sensor_name
        self.logger.debug(f"receive one image for {sensor_name}")
        # self.logger.info(f"step {self.steps[sensor_name]}")
        if self.steps[sensor_name] % self.update_interval == 0 or time.time() - self.last_update_times[sensor_name] > self.time_threshold:
            def _update():
                self.logger.info(f"Updating parameter for {sensor_name}")
                start = time.time()
                encoded_image = data.encoded_image
                image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
                warp_matrix = self.update_warp_matrix(image, sensor_name)
                self.update_warp_matrix_in_redis(warp_matrix, sensor_name)
                self.logger.info(f"Updated warp matrix for sensor: {sensor_name} in {time.time() - start:.2f} seconds")
            x = threading.Thread(target=_update)
            x.setDaemon(True)
            x.start()
        self.steps[sensor_name] += 1
        self.steps[sensor_name] = self.steps[sensor_name] % self.update_interval
        self.last_update_times[sensor_name] = time.time()
