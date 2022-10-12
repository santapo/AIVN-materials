import logging
from typing import Callable, List

import cv2
import numpy as np

from .base_tracker import BaseTracker

logger = logging.getLogger()


class HOGTracker(BaseTracker):
    def __init__(self, image: np.ndarray, roi_window: List[int], tracker: Callable):
        super().__init__(image, roi_window, tracker)
        logger.info(f"Init {self.__class__.__name__} Successfully!")

    def update_roi_feature(self, image: np.ndarray):
        return super().update_roi_feature(image)

    def get_probability_map(self, image: np.ndarray) -> np.ndarray:
        return super().get_probability_map(image)