import torch
import torch.nn as nn

def compute_giou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute Generalized IoU Loss
    
    Args:
        pred_boxes: [N, 4] in format [x_center, y_center, width, height] (normalized)
        target_boxes: [N, 4] in format [x_center, y_center, width, height] (normalized)
    
    Returns:
        giou_loss: Scalar loss value
    """
    # Convert to [x1, y1, x2, y2]
    def center_to_corners(boxes):
        x_c, y_c, w, h = boxes.unbind(-1)
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    pred_boxes_corners = center_to_corners(pred_boxes)
    target_boxes_corners = center_to_corners(target_boxes)
    
    # Intersection
    x1_inter = torch.max(pred_boxes_corners[:, 0], target_boxes_corners[:, 0])
    y1_inter = torch.max(pred_boxes_corners[:, 1], target_boxes_corners[:, 1])
    x2_inter = torch.min(pred_boxes_corners[:, 2], target_boxes_corners[:, 2])
    y2_inter = torch.min(pred_boxes_corners[:, 3], target_boxes_corners[:, 3])
    
    inter_area = (x2_inter - x1_inter).clamp(min=0) * (y2_inter - y1_inter).clamp(min=0)
    
    # Areas
    pred_area = (pred_boxes_corners[:, 2] - pred_boxes_corners[:, 0]) * \
                (pred_boxes_corners[:, 3] - pred_boxes_corners[:, 1])
    target_area = (target_boxes_corners[:, 2] - target_boxes_corners[:, 0]) * \
                  (target_boxes_corners[:, 3] - target_boxes_corners[:, 1])
    
    # Union
    union_area = pred_area + target_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-7)
    
    # Enclosing box
    x1_enclosing = torch.min(pred_boxes_corners[:, 0], target_boxes_corners[:, 0])
    y1_enclosing = torch.min(pred_boxes_corners[:, 1], target_boxes_corners[:, 1])
    x2_enclosing = torch.max(pred_boxes_corners[:, 2], target_boxes_corners[:, 2])
    y2_enclosing = torch.max(pred_boxes_corners[:, 3], target_boxes_corners[:, 3])
    
    enclosing_area = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing)
    
    # GIoU
    giou = iou - (enclosing_area - union_area) / (enclosing_area + 1e-7)
    
    # Loss (1 - GIoU)
    giou_loss = (1 - giou).mean()
    
    return giou_loss