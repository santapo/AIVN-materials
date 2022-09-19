from typing import List, Union



def compute_iou(bbox1: List[Union[int, float]], bbox2: List[Union[int, float]]) -> float:
    """Compute Intersection Over Union

    Args:
        bbox1 : (x, y, w, h) bounding box
        bbox2 : (x, y, w, h) bounding box

    Returns:
        float: IOU value
    """

    # assert all(type(ele) in [int, float] for ele in bbox1), "bbox1 coordinates must be numbers"
    # assert all(type(ele) in [int, float] for ele in bbox2), "bbox2 coordinates must be numbers"
    assert all(ele >= 0 for ele in bbox1), f"bbox1 coordinates must be positive, but recieved {bbox1}"
    assert all(ele >= 0 for ele in bbox2), f"bbox2 coordinates must be positive, but recieved {bbox2}"

    # convert (x,y,w,h) to (x1,y1,x2,y2)
    bbox1_x1 = bbox1[0]
    bbox1_y1 = bbox1[1]
    bbox1_x2 = bbox1[0] + bbox1[2]
    bbox1_y2 = bbox1[1] + bbox1[3]
    bbox2_x1 = bbox2[0]
    bbox2_y1 = bbox2[1]
    bbox2_x2 = bbox2[0] + bbox2[2]
    bbox2_y2 = bbox2[1] + bbox2[3]

    x1 = max(bbox1_x1, bbox2_x1)
    y1 = max(bbox1_y1, bbox2_y1)
    x2 = min(bbox1_x2, bbox2_x2)
    y2 = min(bbox1_y2, bbox2_y2)

    intersection_area = (x2-x1)*(y2-y1)
    intersection_area = intersection_area if intersection_area > 0 else 0
    if intersection_area == 0:
        return 0
    union_area = bbox1[2]*bbox1[3] + bbox2[2]*bbox2[3] - intersection_area
    return intersection_area / union_area

