import logging
from typing import Callable, List

import cv2
import numpy as np

from . import utils
from .base_tracker import BaseTracker

logger = logging.getLogger()


class HOUGHTracker(BaseTracker):
    def __init__(self, image: np.ndarray, roi_window: List[int], tracker: Callable):
        super().__init__(image, roi_window, tracker)
        logger.info(f"Init {self.__class__.__name__} Successfully!")

    def update_roi_feature(self, image: np.ndarray):
        roi_image = image[self.current_window[1]: self.current_window[1] + self.current_window[3],
                          self.current_window[0]: self.current_window[0] + self.current_window[2]]
        hsv_roi_image =  cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        grey_roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        
        self.RT = utils.build_r_table(grey_roi_image)

        mask = cv2.inRange(hsv_roi_image, np.array((0.,30.,20.)), np.array((180.,255.,235.)))
        self.roi_hist = cv2.calcHist([hsv_roi_image],[0],mask,[180],[0,180])
        self.roi_hist = cv2.normalize(self.roi_hist, None, 0, 255, cv2.NORM_MINMAX)

    def get_probability_map(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0,180], 1)
        x, y, w, h = self.get_current_window()
        m_dst = utils.f_dst_weights(image, y, x, h, w)
        tmp = dst * m_dst
        tmp = tmp.astype('uint8')
        image_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        self.probability_map = utils.transform_hough(image_g, self.RT)
        return self.probability_map
