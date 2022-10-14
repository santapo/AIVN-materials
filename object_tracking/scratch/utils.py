from typing import List


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def l2_distance(n: List[float], m: List[float]) -> float:
    if len(n) != len(m):
        raise "Inputs must have equal length"
    return sum((y-x)**2 for x, y in zip(n, m)) ** 0.5