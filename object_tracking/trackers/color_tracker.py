import logging
from typing import Callable, List

import cv2
import numpy as np

from .base_tracker import BaseTracker

logger = logging.getLogger()


class ColorTracker(BaseTracker):
    def __init__(self, image: np.ndarray, roi_window: List[int], tracker: Callable):
        super().__init__(image, roi_window, tracker)
        logger.info(f"Init {self.__class__.__name__} Successfully!")

    def update_roi_feature(self, image: np.ndarray):
        roi_image = image[self.current_window[1]: self.current_window[1] + self.current_window[3],
                          self.current_window[0]: self.current_window[0] + self.current_window[2]]
        roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(roi_image, np.array((0., 60., 32.)), np.array((180., 255., 255)))
        self.roi_hist = cv2.calcHist([roi_image], [0], mask, [180], [0, 180])
        self.roi_hist = cv2.normalize(self.roi_hist, None, 0, 255, cv2.NORM_MINMAX)

    def get_probability_map(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.probability_map = cv2.calcBackProject([image], [0], self.roi_hist, [0, 180], 1)
        return self.probability_map


class NumpyColorTracker(BaseTracker):
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

    def get_probability_map(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.probability_map = self.numpy_backproject(image, self.roi_image)
        return self.probability_map