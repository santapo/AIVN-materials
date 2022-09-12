from typing import List

import cv2
import numpy as np

import logging

logger = logging.getLogger()


class Tracker:
    def __init__(self, *args, **kwargs):
        ...

    def track(self, frame: np.ndarray, window: List[int]):
        raise NotImplementedError


class ColorMSTracker(Tracker):
    def __init__(self, tracked_window, term_crit=None):
        super(ColorMSTracker, self).__init__()
        
        hsv_roi = cv2.cvtColor(tracked_window, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255)))
        self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        self.roi_hist = cv2.normalize(self.roi_hist, None, 0, 255, cv2.NORM_MINMAX)

        self.term_crit = term_crit
        logger.info(f"Init {self.__class__.__name__} Successfully!")

    def track(self, frame: np.ndarray, window: List[int]):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        # apply meanshift to get the new location
        _, shifted_window = cv2.meanShift(dst, window, self.term_crit)
        return shifted_window