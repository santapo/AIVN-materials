import logging
from typing import Callable, List

import cv2
import numpy as np

from .base_tracker import BaseTracker

logger = logging.getLogger()


class SIFTTracker(BaseTracker):
    ...