import torch 
import numpy as np
from typing import List, Dict, Tuple

def calculate_iou_batch(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU between two sets of boxes
    
    Args:
        boxes1: [N, 4] in format [x_center, y_center, width, height] (normalized)
        boxes2: [M, 4] in format [x_center, y_center, width, height] (normalized)
    
    Returns:
        iou_matrix: [N, M] IoU matrix
    """
    def center_to_corners(boxes: torch.Tensor) -> torch.Tensor:
        x_center, y_center, width, height = boxes.unbind(-1)
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2 
        y2 = y_center + height / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    boxes1_corners = center_to_corners(boxes1)  # [N, 4]
    boxes2_corners = center_to_corners(boxes2)  # [M, 4]
    
    # Compute intersections
    x1_max = torch.max(boxes1_corners[:, None, 0], boxes2_corners[None, :, 0])
    y1_max = torch.max(boxes1_corners[:, None, 1], boxes2_corners[None, :, 1])
    x2_min = torch.min(boxes1_corners[:, None, 2], boxes2_corners[None, :, 2])
    y2_min = torch.min(boxes1_corners[:, None, 3], boxes2_corners[None, :, 3])
    
    intersection_width = (x2_min - x1_max).clamp(min=0)
    intersection_height = (y2_min - y1_max).clamp(min=0)
    intersection_area = intersection_width * intersection_height  # [N, M]
    
    # Compute areas
    boxes1_area = (boxes1_corners[:, 2] - boxes1_corners[:, 0]) * (boxes1_corners[:, 3] - boxes1_corners[:, 1])  # [N]
    boxes2_area = (boxes2_corners[:, 2] - boxes2_corners[:, 0]) * (boxes2_corners[:, 3] - boxes2_corners[:, 1])  # [M]
    
    union_area = boxes1_area[:, None] + boxes2_area[None, :] - intersection_area  # [N, M]
    
    # Compute IoU
    iou = intersection_area / union_area.clamp(min=1e-6)
    
    return iou

def apply_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
    """
    Apply Non-Maximum Suppression
    
    Args:
        boxes: [N, 4] in format [x1, y1, x2, y2]
        scores: [N] confidence scores
        iou_threshold: IoU threshold for suppression
    
    Returns:
        keep_indices: List of indices to keep
    """
    if len(boxes) == 0:
        return []
    
    # Convert to torch
    boxes = torch.from_numpy(boxes).float()
    scores = torch.from_numpy(scores).float()
    
    # Get coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Compute areas
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by scores
    _, order = scores.sort(descending=True)
    
    keep = []
    
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        
        i = order[0].item()
        keep.append(i)
        
        # Compute IoU with remaining boxes
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])
        
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Keep boxes with IoU <= threshold
        mask = iou <= iou_threshold
        order = order[1:][mask]
    
    return keep

def compute_ap(
    predictions: List[Dict],
    targets: List[Dict],
    class_id: int,
    iou_threshold: float = 0.5
) -> float:
    """
    Compute Average Precision for a single class
    
    Args:
        predictions: List of prediction dicts with 'boxes', 'labels', 'scores'
        targets: List of target dicts with 'boxes', 'labels'
        class_id: Class ID to compute AP for
        iou_threshold: IoU threshold for matching
    
    Returns:
        ap: Average Precision score
    """
    # Collect all predictions and targets for this class
    all_pred_boxes = []
    all_pred_scores = []
    all_target_boxes = []
    
    for pred, target in zip(predictions, targets):
        # Get predictions for this class
        if len(pred['labels']) > 0:
            class_mask = pred['labels'] == class_id
            if class_mask.any():
                all_pred_boxes.append(pred['boxes'][class_mask])
                all_pred_scores.append(pred['scores'][class_mask])
        
        # Get targets for this class
        if len(target['labels']) > 0:
            class_mask = target['labels'] == class_id
            if class_mask.any():
                all_target_boxes.append(target['boxes'][class_mask])
    
    if len(all_pred_boxes) == 0 or len(all_target_boxes) == 0:
        return 0.0
    
    # Concatenate
    pred_boxes = torch.cat(all_pred_boxes, dim=0)
    pred_scores = torch.cat(all_pred_scores, dim=0)
    
    num_targets = sum(len(boxes) for boxes in all_target_boxes)
    
    # Sort by score
    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    
    # Match predictions to targets
    tp = torch.zeros(len(pred_boxes))
    fp = torch.zeros(len(pred_boxes))
    
    target_matched = []
    for target_boxes in all_target_boxes:
        target_matched.append(torch.zeros(len(target_boxes), dtype=torch.bool))
    
    target_idx = 0
    pred_idx = 0
    
    for i, pred_box in enumerate(pred_boxes):
        best_iou = 0.0
        best_target_idx = -1
        best_target_group = -1
        
        # Find best matching target
        for group_idx, target_boxes in enumerate(all_target_boxes):
            if len(target_boxes) == 0:
                continue
            
            ious = calculate_iou_batch(pred_box.unsqueeze(0), target_boxes)[0]
            max_iou, max_idx = ious.max(0)
            
            if max_iou > best_iou and not target_matched[group_idx][max_idx]:
                best_iou = max_iou.item()
                best_target_idx = max_idx.item()
                best_target_group = group_idx
        
        if best_iou >= iou_threshold:
            tp[i] = 1
            target_matched[best_target_group][best_target_idx] = True
        else:
            fp[i] = 1
    
    # Compute precision and recall
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    
    recalls = tp_cumsum / num_targets
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    # Compute AP (area under PR curve)
    # Use 11-point interpolation
    ap = 0.0
    for t in torch.linspace(0, 1, 11):
        mask = recalls >= t
        if mask.any():
            p = precisions[mask].max()
            ap += p / 11.0
    
    return ap.item()


def compute_map(
    predictions: List[Dict],
    targets: List[Dict],
    num_classes: int,
    iou_threshold: float = 0.5
) -> float:
    """
    Compute mean Average Precision across all classes
    
    Args:
        predictions: List of prediction dicts
        targets: List of target dicts
        num_classes: Number of classes
        iou_threshold: IoU threshold
    
    Returns:
        mAP: Mean Average Precision
    """
    aps = []
    
    for class_id in range(num_classes):
        ap = compute_ap(predictions, targets, class_id, iou_threshold)
        aps.append(ap)
    
    # Average (ignore NaN)
    aps = np.array(aps)
    valid_aps = aps[~np.isnan(aps)]
    
    if len(valid_aps) == 0:
        return 0.0
    
    return float(valid_aps.mean())


def compute_per_class_ap(
    predictions: List[Dict],
    targets: List[Dict],
    num_classes: int,
    iou_threshold: float = 0.5
) -> List[float]:
    """
    Compute AP for each class separately
    
    Returns:
        List of AP scores per class
    """
    aps = []
    
    for class_id in range(num_classes):
        ap = compute_ap(predictions, targets, class_id, iou_threshold)
        aps.append(ap)
    
    return aps


def compute_retrieval_metrics(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute retrieval metrics (Recall@K) for CLIP
    
    Args:
        image_embeds: [N, D] normalized image embeddings
        text_embeds: [N, D] normalized text embeddings
        k_values: List of K values for Recall@K
    
    Returns:
        Dictionary with i2t_R@K and t2i_R@K metrics
    """
    # Normalize
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    # Similarity matrix
    similarity = image_embeds @ text_embeds.t()  # [N, N]
    
    metrics = {}
    
    # Image-to-Text Retrieval
    for k in k_values:
        _, topk_indices = similarity.topk(k, dim=1)
        correct = topk_indices == torch.arange(len(image_embeds)).unsqueeze(1).to(topk_indices.device)
        recall_at_k = correct.any(dim=1).float().mean().item()
        metrics[f'i2t_R@{k}'] = recall_at_k
    
    # Text-to-Image Retrieval
    similarity_t = similarity.t()
    for k in k_values:
        _, topk_indices = similarity_t.topk(k, dim=1)
        correct = topk_indices == torch.arange(len(text_embeds)).unsqueeze(1).to(topk_indices.device)
        recall_at_k = correct.any(dim=1).float().mean().item()
        metrics[f't2i_R@{k}'] = recall_at_k
    
    return metrics