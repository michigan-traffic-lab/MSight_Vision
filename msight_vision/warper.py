import cv2
import numpy as np
from threading import Thread


class ClassicWarper:
    """
    warper class warp image to the standard image
    """

    def __init__(self, standard_img, 
                 starting_behavior="return" # "wait" or "return"
                 ):
        self.standard_img = standard_img
        self.warp_matrix = None
        self.step = 0
        self.update_interval = 1000
        self.starting_behavior = starting_behavior  # "wait" or "return"

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

    def update_warp_matrix(self, image):
        self.warp_matrix = self.get_warp_matrix_between_two_image(
            self.standard_img, image)

    def warp(self, image):
        if self.warp_matrix is None:
            print("WARNING: Warp matrix is not set yet. ")
            if self.starting_behavior == "wait":
                self.update_warp_matrix(image)
            elif self.starting_behavior == "return":
                t = Thread(target=self.update_warp_matrix, args=(image,))
                t.daemon = True  # Set the thread as a daemon thread
                t.start()
                return image
            else:
                raise ValueError("Invalid starting behavior. Use 'wait' or 'return'.")
        # Apply the warp matrix to the image
        sz = image.shape
        new_image = cv2.warpPerspective(
            image, self.warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        self.step += 1
        if self.step % self.update_interval == 0:
            t = Thread(target=self.update_warp_matrix, args=(image,))
            t.daemon = True  # Set the thread as a daemon thread
            t.start()
        return new_image
    
class ClassicWarperWithExternalUpdate:
    def warp(self, image, warp_matrix):
        if warp_matrix is None:
            return image
        # Apply the warp matrix to the image
        sz = image.shape
        new_image = cv2.warpPerspective(
            image, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return new_image
