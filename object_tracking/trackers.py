import logging
from typing import Callable, List

import cv2
import numpy as np

logger = logging.getLogger()


class BaseTracker:
    def __init__(self, image: np.ndarray[np.int8], roi_window: List[int], tracker: Callable):
        self.update_current_window(roi_window)
        self.update_roi_feature(image)
        self.tracker = tracker

    def update_roi_feature(self, image: np.ndarray[np.int8]):
        raise NotImplementedError

    def update_current_window(self, window: List[int]):
        self.current_window = window

    def get_current_window(self) -> List[int]:
        return self.current_window

    def get_probability_map(self, image: np.ndarray[np.int8]) -> np.ndarray[np.int8]:
        self.probability_map = None
        raise NotImplementedError

    def track(self, frame: np.ndarray):
        self.get_probability_map(frame)
        _, new_window = self.tracker(self.probability_map, self.current_window)
        self.update_current_window(new_window)


class ColorTracker(BaseTracker):
    def __init__(self, image: np.ndarray[np.int8], roi_window: List[int], tracker: Callable):
        super(ColorTracker, self).__init__(image, roi_window, tracker)
        logger.info(f"Init {self.__class__.__name__} Successfully!")

    def update_roi_feature(self, image: np.ndarray[np.int8]):
        roi_image = image[self.current_window[1]: self.current_window[1] + self.current_window[3],
                          self.current_window[0]: self.current_window[0] + self.current_window[2]]
        roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(roi_image, np.array((0., 60., 32.)), np.array((180., 255., 255)))
        self.roi_hist = cv2.calcHist([roi_image], [0], mask, [180], [0, 180])
        self.roi_hist = cv2.normalize(self.roi_hist, None, 0, 255, cv2.NORM_MINMAX)

    def get_probability_map(self, image: np.ndarray[np.int8]) -> np.ndarray[np.int8]:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        probability_map = cv2.calcBackProject([image], [0], self.roi_hist, [0, 180], 1)
        return probability_map


class HOGTracker(BaseTracker):
    ...

class SIFTTracker(BaseTracker):
    ...