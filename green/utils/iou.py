import torch


def intersection_over_union(bboxes_preds, bboxes_gt, bbox_format="corner_point"):
    """ Calculate intersection over union
    Args:
        bboxes_preds:   (N, 4) [[x1, y1, w1, h1]]
        bboxes_gt:      (N, 4)

    """

    if bbox_format == "corner_point":
        bbox1_x1 = bboxes_preds[..., 0:1]
        bbox1_y1 = bboxes_preds[..., 1:2]
        bbox1_x2 = bboxes_preds[..., 0:1] + bboxes_preds[..., 2:3]
        bbox1_y2 = bboxes_preds[..., 1:2] + bboxes_preds[..., 3:4]
        bbox2_x1 = bboxes_gt[..., 0:1]
        bbox2_y1 = bboxes_gt[..., 1:2]
        bbox2_x2 = bboxes_gt[..., 0:1] + bboxes_gt[..., 2:3]
        bbox2_y2 = bboxes_gt[..., 1:2] + bboxes_gt[..., 3:4]

    if bbox_format == "mid_point":
        raise ...
    
    x1 = torch.max(bbox1_x1, bbox2_x1)
    y1 = torch.max(bbox1_y1, bbox2_y1)
    x2 = torch.min(bbox1_x2, bbox2_x2)
    y2 = torch.min(bbox1_y2, bbox2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    bbox1_area = bboxes_preds[..., 2:3] * bboxes_preds[..., 3:4]
    bbox2_area = bboxes_gt[..., 2:3] * bboxes_gt[..., 3:4]

    iou = intersection / (bbox1_area + bbox2_area - intersection + 1e-8)

    return iou

if __name__ == "__main__":
    import torch

    bbox_pred = torch.tensor([[2, 3, 20, 20]])
    bbox_gt = torch.tensor([[2, 3, 20, 20]])

    iou = intersection_over_union(bbox_pred, bbox_gt, bbox_format="corner_point")

    print(iou)



