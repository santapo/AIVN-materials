import math
from typing import List

import numpy as np


def meanshift(prob_image: np.ndarray[np.int8], window: List[int]) -> List[int]:
    x, y, w, h = window

    M00, M01, M10 = 0, 0, 0
    for i in range(x, x + w):
        for j in range(y, y + h):
            M00 = M00 + prob_image[i, j]
            M10 = M10 + prob_image[i, j] * j
            M01 = M01 + prob_image[i, j] * i

    xc = M01 / M00
    if xc + w/2 > prob_image.shape[0]:
        xc = prob_image.shape[0] - w/2
    yc = M10 / M00
    if yc + h/2 > prob_image.shape[1]:
        yc = prob_image.shape[1] - h/2

    new_x = math.floor(xc) - w/2
    new_y = math.floor(yc) - h/2
    return None, [new_x, new_y, w, h]