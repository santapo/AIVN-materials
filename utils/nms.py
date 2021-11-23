



from iou import intersection_over_union


def non_maximum_suppression(bboxes, iou_threshold, class_threshold, bbox_format="corner_point"):
    """[summary]

    Args:
        bboxes ([type]): List of Torch Tensors, [[class_score, x, y, w, h]]
        iou_threshold ([type]): [description]
        class_threshold ([type]): [description]
        bbox_format (str, optional): [description]. Defaults to "corner_point".
    """

    bboxes = [box for box in bboxes if box[0] > class_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)

    result_bboxes = []

    while bboxes:
        best_bbox = bboxes.pop(0) # get best box and remove it from bboxes list
        bboxes = [box for box in bboxes if intersection_over_union(best_bbox[1:], box[1:], bbox_format=bbox_format) < iou_threshold] # remove overlapped bboxes by iou_threshold
        result_bboxes.append(best_bbox)

    return result_bboxes

if __name__ == "__main__":
    import torch

    bboxes = [
        torch.tensor([0.9, 2, 3, 20, 20]),
        torch.tensor([0.5, 5, 6, 20, 20]),
        torch.tensor([0.7, 16, 17, 20, 20]),
        torch.tensor([0.6, 40, 41, 20, 20]),
        torch.tensor([0.8, 46, 47, 20, 20]),
    ]

    result_bboxes = non_maximum_suppression(bboxes, 0.7, 0.7)
    print(result_bboxes)
