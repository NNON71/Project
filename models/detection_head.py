import torch
import torch.nn as nn
from typing import Optional, Tuple

class OWLViTClassificationHead(nn.Module) :
    """Classification head for OWL-ViT model.

    Args:
        nn (Module): PyTorch neural network module.
    """
    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        num_layers: int = 3,
        dropout: float = 0.1
    ) :
        """ Initializes the OWLViTClassificationHead.

        Args:
            vision_hidden_size (int): dimension of vision hidden size.
            text_hidden_size (int): dimension of text hidden size.
            num_layer (int, optional): number of layers. Defaults to 3.
            dropout (float, optional): dropout rate. Defaults to 0.1.
        """
        super().__init__()
        
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
       
        # Vision feature processing
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_hidden_size, vision_hidden_size),
            nn.LayerNorm(vision_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Text query processing
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden_size, text_hidden_size),
            nn.LayerNorm(text_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Classification laters
        layers = []
        input_dim = vision_hidden_size
        for _ in range(num_layers - 1) :
            layers.extend([
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            
        self.classifier = nn.Sequential(*layers)
        
    def forward(
        self,
        image_features: torch.Tensor,
        query_embeds: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor] :
        """
        Args:
            image_feats: [B, num_patches, vision_hidden_size]
            query_embeds: [B, num_queries, text_hidden_size] (optional)
            query_mask: [B, num_queries] (optional)
        
        Returns:
            pred_logits: [B, num_patches, num_queries]
            class_embeds: [B, num_queries, text_hidden_size]

        """
        
        batch_size, num_patches, _ = image_features.shape
        
        # Process vision features
        class_embeds = self.vision_proj(image_features)  # [B, num_patches, vision_hidden_size]
        class_embeds = self.classifier(class_embeds)  # [B, num_patches, vision_hidden_size]
        
        if query_embeds is not None:
            # Process text queries
            query_embeds = self.text_proj(query_embeds)  # [B, num_queries, D_text]
            
            # ถ้า dimension ไม่ตรง ให้ project
            if query_embeds.shape[-1] != class_embeds.shape[-1]:
                query_proj = nn.Linear(
                    query_embeds.shape[-1],
                    class_embeds.shape[-1],
                    bias=False
                ).to(query_embeds.device)
                query_embeds = query_proj(query_embeds)
            
            # Normalize embeddings
            class_embeds_norm = class_embeds / (class_embeds.norm(dim=-1, keepdim=True) + 1e-6)
            query_embeds_norm = query_embeds / (query_embeds.norm(dim=-1, keepdim=True) + 1e-6)
            
            # Compute similarity (dot product)
            # [B, num_patches, D] @ [B, D, num_queries] = [B, num_patches, num_queries]
            pred_logits = torch.einsum('bnd,bqd->bnq', class_embeds_norm, query_embeds_norm)
            
            # Scale logits
            pred_logits = pred_logits * 100.0  # Temperature scaling
            
            # Apply query mask if provided
            if query_mask is not None:
                # [B, 1, num_queries]
                query_mask = query_mask.unsqueeze(1)
                pred_logits = pred_logits.masked_fill(~query_mask, float('-inf'))
        else:
            # No queries provided, return zeros
            num_queries = 1
            pred_logits = torch.zeros(
                batch_size, num_patches, num_queries,
                device=image_features.device
            )
        
        return pred_logits, class_embeds

class OWLViTBoxPredictionHead(nn.Module) :
    """Box prediction head for OWL-ViT model.

    Args:
        nn (Module): PyTorch neural network module.
    """
    def __init__(
        self,
        vision_hidden_size: int,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Args:
            vision_hidden_size (int): dimension of vision hidden size.
            num_layers (int, optional): number of layers. Defaults to 3.
            dropout (float, optional): dropout rate. Defaults to 0.1.
        """
        super().__init__()
        
        self.vision_hidden_size = vision_hidden_size
        
        # Box regression MLP
        layers = []
        in_dim = vision_hidden_size
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, in_dim),
                nn.LayerNorm(in_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        
        # Final layer outputs 4 coordinates
        layers.append(nn.Linear(in_dim, 4))
        
        self.box_regressor = nn.Sequential(*layers)
        
        # Initialize last layer with small values
        nn.init.constant_(self.box_regressor[-1].bias, 0)
        nn.init.uniform_(self.box_regressor[-1].weight, -0.001, 0.001)
        
        print(f"✓ Box Prediction Head initialized")
        print(f"  Vision dim: {vision_hidden_size}")
        print(f"  Layers: {num_layers}")
        print(f"  Output: 4 (normalized coordinates)")
    
    def forward(self, image_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_feats: [B, num_patches, vision_hidden_size]
        
        Returns:
            pred_boxes: [B, num_patches, 4]
                4 = [x_center, y_center, width, height] (normalized [0, 1])
        """
        # Predict boxes
        pred_boxes = self.box_regressor(image_feats)  # [B, num_patches, 4]
        
        # Apply sigmoid to ensure [0, 1] range
        pred_boxes = torch.sigmoid(pred_boxes)
        
        return pred_boxes
    
# =================== Test Code ===================

# print("Testing OWLViTDetectionHead...")

# batch_size = 2
# num_patches = 49 # 7x7 patches
# vision_hidden_size = 768
# text_hidden_size = 768
# num_queries = 5

# image_feats = torch.randn(batch_size, num_patches, vision_hidden_size)
# query_embeds = torch.randn(batch_size, num_queries, text_hidden_size)
# query_mask = torch.ones(batch_size, num_queries, dtype=torch.bool)

# print("\n" + "="*60)
# print("Testing Classification Head")
# print("="*60)

# class_head = OWLViTClassificationHead(
#     vision_hidden_size=vision_hidden_size,
#     text_hidden_size=text_hidden_size,
#     num_layers=3
# )

# pred_logits, class_embeds = class_head(
#     image_features=image_feats,
#     query_embeds=query_embeds,
#     query_mask=query_mask
# )

# print(f"\nInput shapes:")
# print(f"  Image features: {image_feats.shape}")
# print(f"  Query embeddings: {query_embeds.shape}")
# print(f"\nOutput shapes:")
# print(f"  Prediction logits: {pred_logits.shape}")
# print(f"  Class embeddings: {class_embeds.shape}")

# # Test Box Prediction Head
# print("\n" + "="*60)
# print("Testing Box Prediction Head")
# print("="*60)

# box_head = OWLViTBoxPredictionHead(
#     vision_hidden_size=vision_hidden_size,
#     num_layers=3
# )

# pred_boxes = box_head(image_feats=image_feats)

# print(f"\nInput shape:")
# print(f"  Image features: {image_feats.shape}")
# print(f"\nOutput shape:")
# print(f"  Predicted boxes: {pred_boxes.shape}")
# print(f"\nBox coordinate ranges:")
# print(f"  Min: {pred_boxes.min().item():.3f}")
# print(f"  Max: {pred_boxes.max().item():.3f}")
# print(f"  Mean: {pred_boxes.mean().item():.3f}")

# # Count parameters
# total_params_class = sum(p.numel() for p in class_head.parameters())
# total_params_box = sum(p.numel() for p in box_head.parameters())

# print(f"\nParameters:")
# print(f"  Classification Head: {total_params_class:,}")
# print(f"  Box Prediction Head: {total_params_box:,}")
# print(f"  Total: {total_params_class + total_params_box:,}")

"""
============================================================
Testing Classification Head
============================================================

Input shapes:
  Image features: torch.Size([2, 49, 768])
  Query embeddings: torch.Size([2, 5, 768])

Output shapes:
  Prediction logits: torch.Size([2, 49, 5])
  Class embeddings: torch.Size([2, 49, 768])

============================================================
Testing Box Prediction Head
============================================================
✓ Box Prediction Head initialized
  Vision dim: 768
  Layers: 3
  Output: 4 (normalized coordinates)

Input shape:
  Image features: torch.Size([2, 49, 768])

Output shape:
  Predicted boxes: torch.Size([2, 49, 4])

Box coordinate ranges:
  Min: 0.492
  Max: 0.509
  Mean: 0.499

Parameters:
  Classification Head: 2,368,512
  Box Prediction Head: 1,187,332
  Total: 3,555,844
"""