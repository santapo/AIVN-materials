import torch
from collections import Counter

from iou import intersection_over_union


# bbox_pred = (idx, class, class_prob, x, y, w, h)
# bbox_gt = (idx, class, 1, x, y, w, h)
def mean_average_precision(bbox_preds, bbox_gts, iou_threshold, bbox_format, num_classes):

    average_precision = []

    epsilon = 1e-7
    for c in range(num_classes):
        predictions = []
        ground_truths = []

        for pred in bbox_preds:
            if pred[1] == c:
                predictions.append(pred)

        for gt in bbox_gts:
            if gt[1] == c:
                ground_truths.append(gt)

        bbox_counter = Counter([gt[0] for gt in ground_truths])
        for k, v in bbox_counter:
            bbox_counter[k] = torch.zeros(v)

        predictions.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(predictions))
        FP = torch.zeros(len(predictions))
        total_bbox_gt = len(ground_truths)

        if total_bbox_gt == 0:
            continue
        
        for pred_idx, pred in enumerate(predictions):
            single_sample_gt = [bbox for bbox in ground_truths if bbox[0] == pred[0]]

            best_iou = 0.
            for idx, gt in enumerate(single_sample_gt):
                iou = intersection_over_union(
                        torch.tensor(pred[3:]),
                        torch.tensor(gt[3:]),
                        bbox_format=bbox_format)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if bbox_counter[pred[0]][best_gt_idx] == 0:
                    TP[pred_idx] = 1
                    bbox_counter[pred[0]][best_gt_idx] = 1
            else:
                FP[pred_idx] = 1

        TP_sum = torch.cumsum(TP, dim=0)
        FP_sum = torch.cumsum(FP, dim=0)
        precision = TP_sum / (TP_sum + FP_sum + epsilon)
        recall = TP_sum / (total_bbox_gt + epsilon)
        precision = torch.cat(torch.tensor([0]), precision)
        recall = torch.cat(torch.tensor([0]), recall)
        average_precision.append(torch.trapz(precision, recall))

    return sum(average_precision) / len(average_precision)

            


            

