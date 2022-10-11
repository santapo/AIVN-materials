import math
from typing import List

import numpy as np


def meanshift(probImage: np.ndarray[np.int8], window: List[int]) -> List[int]:
    x, y, w, h = window

    M00, M01, M10 = 0, 0, 0
    for i in range(x, x + w):
        for j in range(y, y + h):
            M00 = M00 + probImage[i, j]
            M10 = M10 + probImage[i, j] * j
            M01 = M01 + probImage[i, j] * i

    xc = M01 / M00
    if xc + w/2 > probImage.shape[0]:
        xc = probImage.shape[0] - w/2
    yc = M10 / M00
    if yc + h/2 > probImage.shape[1]:
        yc = probImage.shape[1] - h/2

    new_window = [math.floor(xc) - w/2, math.floor(yc) - h/2, w, h]
    return new_window