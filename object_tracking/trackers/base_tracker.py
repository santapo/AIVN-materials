import logging
from typing import Callable, List

import numpy as np

logger = logging.getLogger()


class BaseTracker:
    def __init__(self, image: np.ndarray, roi_window: List[int], tracker: Callable):
        self.update_current_window(roi_window)
        self.update_roi_feature(image)
        self.tracker = tracker

    def update_roi_feature(self, image: np.ndarray):
        raise NotImplementedError

    def update_current_window(self, window: List[int]):
        self.current_window = window

    def get_current_window(self) -> List[int]:
        return self.current_window

    def get_probability_map(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def track(self, frame: np.ndarray):
        self.get_probability_map(frame)
        _, new_window = self.tracker(self.probability_map, self.current_window)
        self.update_current_window(new_window)