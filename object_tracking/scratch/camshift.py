import math
from typing import List

import numpy as np


def camshift(probImage: np.ndarray[np.int8], window: List[int]) -> List[int]:
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
    
    new_x = math.floor(xc) - w/2
    new_y = math.floor(yc) - h/2
    new_w = math.floor(2 * math.sqrt(M00/256))
    new_h = 1.2 * new_w
    return new_x, new_y, new_w, new_h