import logging
from typing import Callable, List

import cv2
import numpy as np

from .utils import get_2d_mix_gaussian

from .base_tracker import BaseTracker

logger = logging.getLogger()


class SIFTTracker(BaseTracker):
    def __init__(self, image: np.ndarray, roi_window: List[int], tracker: Callable):
        super().__init__(image, roi_window, tracker)
        logger.info(f"Init {self.__class__.__name__} Successfully!")

    @staticmethod
    def numpy_backproject(hsv_image: np.ndarray, hsv_roi_image: np.ndarray) -> np.ndarray:
        M = cv2.calcHist([hsv_roi_image], [0, 1], None, [180, 256], [0, 180, 0, 256])
        I = cv2.calcHist([hsv_image], [0, 1], None, [180, 256], [0, 180, 0, 256])
        R = M / I
        h, s, v = cv2.split(hsv_image)
        B = R[h.ravel(), s.ravel()]
        B = np.minimum(B, 1)
        B = B.reshape(hsv_image.shape[:2])
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cv2.filter2D(B, -1, disc, B)
        B = np.uint8(B)
        return B

    def update_roi_feature(self, image: np.ndarray):
        roi_image = image[self.current_window[1]: self.current_window[1] + self.current_window[3],
                          self.current_window[0]: self.current_window[0] + self.current_window[2]]
        self.roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)

    def get_sift_map(self, image: np.ndarray) -> np.ndarray:
        sift = cv2.SIFT_create()
        kp_tgt, des_tgt = sift.detectAndCompute(image, None)
        kp_query, des_query = sift.detectAndCompute(self.roi_image, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des_query, des_tgt, k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)

        center_list = [kp_tgt[m.trainIdx].pt for m in good]
        radius_list = [kp_tgt[m.trainIdx].size for m in good]
        self.sift_map = get_2d_mix_gaussian(image.shape[:2], radius_list, center_list)
        return self.sift_map

    def get_probability_map(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.color_probability_map = self.numpy_backproject(image, self.roi_image)
        self.probability_map = self.color_probability_map + self.get_sift_map(image) * 255
        return self.probability_map