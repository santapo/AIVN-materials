import logging
from typing import Callable, List

import cv2
import numpy as np
from .utils import l2_distance
from .base_tracker import BaseTracker

logger = logging.getLogger()


class AdaTracker(BaseTracker):
    def __init__(self, image: np.ndarray, roi_window: List[int], track_core: Callable, scale_range: List[int]):
        super().__init__(image, roi_window, track_core)
        self.scale_range = scale_range
        self.scale_arr = np.arange(*self.scale_range)
        self.update_current_scale(scale=1)
        logger.info(f"Init {self.__class__.__name__} Successfully!")

    def update_roi_feature(self, image: np.ndarray):
        self.roi_image = image[self.current_window[1]: self.current_window[1] + self.current_window[3],
                          self.current_window[0]: self.current_window[0] + self.current_window[2]]
        self.roi_image = cv2.cvtColor(self.roi_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(self.roi_image, np.array((0., 60., 32.)), np.array((180., 255., 255)))
        self.roi_hist = cv2.calcHist([self.roi_image], [0], mask, [180], [0, 180])
        self.roi_hist = cv2.normalize(self.roi_hist, None, 0, 255, cv2.NORM_MINMAX)

    def update_current_scale(self, scale: float = 1.):
        self.current_scale = scale

    def get_current_scale(self) -> float:
        return self.current_scale

    def update_current_window(self, window: List[int]):
        # window = [int(i) for i in window]
        self.current_window = window

    def get_probability_map(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.probability_map = cv2.calcBackProject([image], [0], self.roi_hist, [0, 180], scale=1)

        self.scale_probability_map = []
        for i, scale in enumerate(self.scale_arr):
            back_project_scale = 1.1**scale
            self.scale_probability_map.append(cv2.calcBackProject([image], [0], self.roi_hist,[0, 180], scale=back_project_scale))
        self.scale_probability_map = np.array(self.scale_probability_map)
        return self.probability_map

    def track(self, frame: np.ndarray, max_iters: int = 5, scale_eps: float = 1e-3, space_eps: float = 5):
        self.get_probability_map(frame)
        for _ in range(max_iters):
            _, new_window = self.track_core(self.probability_map, self.current_window)

            x, y, w, h = [int(i) for i in new_window]
            scale_probability_map = self.scale_probability_map[:, y: y+h//2, x: x+w//2]
            scale_probability_map = np.sum(scale_probability_map, axis=(1, 2))

            new_scale_index = np.argmax(scale_probability_map)
            new_scale = self.scale_arr[new_scale_index]

            self.update_current_scale(new_scale)
            new_window = [x, y, w*self.current_scale, h*self.current_scale]
            self.update_current_window(new_window)