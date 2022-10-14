import math
from typing import List

import numpy as np

from .utils import clamp, l2_distance


def meanshift(prob_image: np.ndarray, window: List[int], max_iters: int = 10, eps: float = 5) -> List[int]:
    for i in range(max_iters):
        x, y, w, h = window
        M00, M01, M10 = 0, 0, 0
        for i in range(x, clamp(x + round(w), x, prob_image.shape[1])):
            for j in range(y, clamp(y + round(h), y, prob_image.shape[0])):
                M00 = M00 + prob_image[j, i]
                M10 = M10 + prob_image[j, i] * j
                M01 = M01 + prob_image[j, i] * i
        xc = M01 / (M00 + 1e-3)
        yc = M10 / (M00 + 1e-3)
        new_x = clamp(int(math.floor(xc) - w/2), 0, prob_image.shape[1] - w/2)
        new_y = clamp(int(math.floor(yc) - h/2), 0, prob_image.shape[0] - h/2)

        window = [new_x, new_y, w, h]
        if l2_distance((new_x, new_y), (x, y)) < eps: break
    return None, window