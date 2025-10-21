import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from typing import List, Optional
import os


def draw_detections(
    image: Image.Image,
    boxes: List[List[float]],
    labels: List[str],
    scores: Optional[List[float]] = None,
    box_color: str = 'red',
    text_color: str = 'white',
    font_size: int = 20
) -> Image.Image:
    """
    Draw bounding boxes and labels on image
    
    Args:
        image: PIL Image
        boxes: List of boxes in [x1, y1, x2, y2] format
        labels: List of label strings (Thai)
        scores: Optional list of confidence scores
        box_color: Color for bounding boxes
        text_color: Color for text
        font_size: Font size for labels
    
    Returns:
        Image with drawn detections
    """
    # Create a copy
    image = image.copy()
    draw = ImageDraw.Draw(image)
    
    # Try to load Thai font
    try:
        font = ImageFont.truetype("THSarabunNew.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/tlwg/Garuda.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Draw each detection
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
        
        # Prepare label text
        if scores is not None:
            text = f"{label}: {scores[i]:.2f}"
        else:
            text = label
        
        # Draw label background
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(text_bbox, fill=box_color)
        
        # Draw label text
        draw.text((x1, y1), text, fill=text_color, font=font)
    
    return image


def visualize_predictions(
    image_tensor: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: Optional[torch.Tensor] = None,
    gt_labels: Optional[torch.Tensor] = None,
    class_names: Optional[List[str]] = None
) -> plt.Figure:
    """
    Visualize predictions vs ground truth using matplotlib
    
    Args:
        image_tensor: [3, H, W] normalized image tensor
        pred_boxes: [N, 4] predicted boxes (normalized)
        pred_labels: [N] predicted labels
        pred_scores: [N] confidence scores
        gt_boxes: [M, 4] ground truth boxes (optional)
        gt_labels: [M] ground truth labels (optional)
        class_names: List of class names
    
    Returns:
        matplotlib Figure
    """
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image_tensor * std + mean
    image = image.permute(1, 2, 0).cpu().numpy()
    image = np.clip(image, 0, 1)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    H, W = image.shape[:2]
    
    # Draw predictions (red)
    for i in range(len(pred_boxes)):
        box = pred_boxes[i].cpu().numpy()
        label = pred_labels[i].item()
        score = pred_scores[i].item()
        
        # Convert from normalized center format to pixel coordinates
        x_c, y_c, w, h = box
        x1 = (x_c - w / 2) * W
        y1 = (y_c - h / 2) * H
        width = w * W
        height = h * H
        
        # Draw box
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        if class_names:
            text = f"{class_names[label]}: {score:.2f}"
        else:
            text = f"Class {label}: {score:.2f}"
        
        ax.text(
            x1, y1 - 5,
            text,
            bbox=dict(facecolor='red', alpha=0.7),
            fontsize=10,
            color='white'
        )
    
    # Draw ground truth (green)
    if gt_boxes is not None and gt_labels is not None:
        for i in range(len(gt_boxes)):
            box = gt_boxes[i].cpu().numpy()
            label = gt_labels[i].item()
            
            x_c, y_c, w, h = box
            x1 = (x_c - w / 2) * W
            y1 = (y_c - h / 2) * H
            width = w * W
            height = h * H
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='green', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
            
            if class_names:
                text = f"GT: {class_names[label]}"
            else:
                text = f"GT: Class {label}"
            
            ax.text(
                x1, y1 + height + 15,
                text,
                bbox=dict(facecolor='green', alpha=0.7),
                fontsize=10,
                color='white'
            )
    
    ax.axis('off')
    plt.tight_layout()
    
    return fig