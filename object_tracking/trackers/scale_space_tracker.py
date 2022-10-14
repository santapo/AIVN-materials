from decimal import DivisionByZero
import logging
from typing import Callable, List

import cv2
import numpy as np
from .utils import l2_distance
from .base_tracker import BaseTracker

logger = logging.getLogger()


class ScaleSpaceTracker(BaseTracker):
    def __init__(self, image: np.ndarray, roi_window: List[int], tracker: Callable, scale_range: List[int]):
        super().__init__(image, roi_window, tracker)
        self.scale_range = scale_range
        self.update_current_scale(scale=1)
        logger.info(f"Init {self.__class__.__name__} Successfully!")

    def update_roi_feature(self, image: np.ndarray):
        roi_image = image[self.current_window[1]: self.current_window[1] + self.current_window[3],
                          self.current_window[0]: self.current_window[0] + self.current_window[2]]
        roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(roi_image, np.array((0., 60., 32.)), np.array((180., 255., 255)))
        self.roi_hist = cv2.calcHist([roi_image], [0], mask, [180], [0, 180])
        self.roi_hist = cv2.normalize(self.roi_hist, None, 0, 255, cv2.NORM_MINMAX)

    def update_current_scale(self, scale: float = 1.):
        self.current_scale = scale

    def get_current_scale(self) -> float:
        return self.current_scale

    def update_current_window(self, window: List[int]):
        window = [int(i) for i in window]
        self.current_window = window

    def get_probability_map(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.probability_map = cv2.calcBackProject([image], [0], self.roi_hist, [0, 180], scale=1)

        self.scale_probability_map = []
        for i, scale in enumerate(range(*self.scale_range)):
            sigma = 1.1**scale
            self.scale_probability_map.append(cv2.calcBackProject([image], [0], self.roi_hist,[0, 180], scale=sigma))
        self.scale_probability_map = np.array(self.scale_probability_map)
        # self.scale_probability_map = np.diff(scale_probability_map, axis=0)
        return self.probability_map

    def track(self, frame: np.ndarray, max_iters: int = 5, scale_eps: float = 1e-3, space_eps: float = 5):
        self.get_probability_map(frame)
        for _ in range(max_iters):
            #  = self.get_current_window()
            # w
            _, new_window = self.tracker(self.probability_map, self.current_window)
            
            x, y, w, h = new_window
            scale_probability_map = self.scale_probability_map[:, y: y+h//2, x: x+w//2]
            scale_probability_map = np.sum(scale_probability_map, axis=(1, 2))
            weighted_scale_probability_map = scale_probability_map * np.array(range(*self.scale_range)) * np.array([0.2, 0.3, 0.2, 0.3, 0.2])
            # import ipdb; ipdb.set_trace()
            new_scale = np.sum(weighted_scale_probability_map) / np.sum(scale_probability_map)
            # new_scale = round(new_scale, 1)
            if np.isnan(new_scale):
                logger.warning("`np.sum(scale_probability_map)` == 0, set `new_scale=1`")
                new_scale = 0
            self.update_current_scale(1.1**new_scale)
            try:
                new_window = [x, y, w*self.current_scale, h*self.current_scale]
            except:
                import ipdb; ipdb.set_trace()
            self.update_current_window(new_window)

            # print(new_window, new_scale)
            # if l2_distance()



