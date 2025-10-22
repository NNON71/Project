import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from models.clip_backbone import OWLViTCLIPBackbone
from models.detection_head import OWLViTClassificationHead, OWLViTBoxPredictionHead

class CustomOWLViTForObjectDetection(nn.Module) :
    """Custom OWL-ViT model for object detection.

    Args:
        nn (Module): PyTorch neural network module.
        
    Architecture: 
        - Backbone : OWLViTCLIPBackbone 
            - Vision Encoder : "clip/clip-vit-base-patch32"
            - Text Encoder : "clicknext/phayathaibert"
        - Detection Head : OWLViTDetectionHead
    """
    def __init__(
        self,
        d_out: int = 512,
        image_encoder_name: str = "openai/clip-vit-base-patch32",
        text_encoder_name: str = "clicknext/phayathaibert",
        freeze_clip: bool = True,
        num_detection_layers: int = 3,
        dropout: float = 0.1
    ) :
        """ Initializes the CustomOWLViTForObjectDetection.

        Args:
            d_out (int, optional): Output dimension. Defaults to 512.
            image_encoder_name (str, optional): Image encoder model name. Defaults to "openai/clip-vit-base-patch32".
            text_encoder_name (str, optional): Text encoder model name. Defaults to "clicknext/phayathaibert".
            freeze_clip (bool, optional): Whether to freeze CLIP encoders. Defaults to True.
            num_detection_layers (int, optional): Number of detection head layers. Defaults to 3.
            detection_dropout (float, optional): Dropout rate for detection head. Defaults to 0.1.
        """
        super().__init__()
        
        # Backbone
        self.backbone = OWLViTCLIPBackbone(
            d_out=d_out,
            image_encoder_name=image_encoder_name,
            text_encoder_name=text_encoder_name,
            freeze_vision=freeze_clip,
            freeze_text=freeze_clip
        )
        
        vision_hidden_size = self.backbone.vision_model.config.hidden_size
        text_hidden_size = self.backbone.text_model.config.hidden_size

        # Classification Head
        self.class_head = OWLViTClassificationHead(
            vision_hidden_size=vision_hidden_size,
            text_hidden_size=text_hidden_size,
            num_layers=num_detection_layers,
            dropout=dropout
        )
        
        self.box_head = OWLViTBoxPredictionHead(
            vision_hidden_size=vision_hidden_size,
            num_layers=num_detection_layers,
            dropout=dropout
        )
        
        # Layer Norm for vision features
        self.layer_norm = nn.LayerNorm(vision_hidden_size, eps=1e-6)
        
        # Text Projection for detection
        self.text_projection_for_detection = nn.Linear(d_out, text_hidden_size, bias=False)
        
        # Store tokenizer
        self.tokenizer = self.backbone.tokenizer
        
        # Store dimensions
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.projection_dim = d_out
        
        # Image Patch Grid Size (img_size: 224x224, patch_size: 32x32)
        self.num_patches_height = 224 // 32
        self.num_patches_width = 224 // 32
        self.num_patches = self.num_patches_height * self.num_patches_width
        
        self._print_model_summary()
        
    def _print_model_summary(self):
        """Print model summary"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("\n" + "="*80)
        print("Model Summary")
        print("="*80)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Frozen Parameters: {total_params - trainable_params:,}")
        print(f"\nDimensions:")
        print(f"  Vision Hidden Size: {self.vision_hidden_size}")
        print(f"  Text Hidden Size: {self.text_hidden_size}")
        print(f"  Projection Dim: {self.projection_dim}")
        print(f"  Num Patches: {self.num_patches} ({self.num_patches_height}x{self.num_patches_width})")
        print("="*80 + "\n")
        
    def image_text_embedder(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None   
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates image and text embeddings.

        Args:
            pixel_values (torch.Tensor): Input image tensor. [B, C, H, W]
            input_ids (torch.Tensor, optional): Input text token IDs. Defaults to None. [B, seq_len]
            attention_mask (torch.Tensor, optional): Attention mask for text. Defaults to None. [B, seq_len]

        Returns:
            text_embeds: [B, projection_dim] or None
            feature_maps: [B, H_patch, W_patch, vision_hidden_size]
            clip_outputs: Dict[str, Any]
        """
        clip_outputs = self.backbone(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Process Vision Features
        vision_outputs = clip_outputs['vision_model_output']
        last_hidden_state = vision_outputs['last_hidden_state']  # [B, num_patches+1, vision_hidden_size]
        
        # Apply layer norm
        image_embeds = self.layer_norm(last_hidden_state)  # [B, num_patches+1, vision_hidden_size]
        
        # Broadcast CLS token to all patches 
        class_token_out = image_embeds[:, :1, :].expand(-1, image_embeds.shape[1]-1, -1) # [B, num_patches, vision_hidden_size]
        
        # Remove CLS token and merge with class token
        image_embeds = image_embeds[:, 1:, :] * class_token_out  # [B, num_patches, vision_hidden_size]
        image_embeds = self.layer_norm(image_embeds)  # [B, num_patches, vision_hidden_size]
        
        # Reshape to feature maps
        batch_size = image_embeds.shape[0]
        feature_map = image_embeds.reshape(
            batch_size,
            self.num_patches_height,
            self.num_patches_width,
            self.vision_hidden_size
        )
        
        # Get text embeddings
        text_embeds = clip_outputs.get('text_embeds', None)  # [B, projection_dim] or None
        
        return text_embeds, feature_map, clip_outputs
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward Pass

        Args:
            pixel_values (torch.Tensor): [B, C, H, W]
            input_ids (Optional[torch.Tensor], optional): [B, seq_len]. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): [B, seq_len]. Defaults to None.
            return_loss (bool, optional): Whether to return loss. Defaults to False.
        Returns:
            {
                'logits': [B, num_patches, num_queries],
                'pred_boxes': [B, num_patches, 4],
                'text_embeds': [B, num_queries, text_hidden_size],
                'image_embeds': [B, H, W, vision_hidden_size],
                'class_embeds': [B, num_patches, vision_hidden_size],
                'clip_loss': scalar (optional)
            }
        """
        
        # Get Embeddings
        text_embeds, feature_map, clip_outputs = self.image_text_embedder(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Flatten feature maps for detection head
        batch_size, h, w, hidden_dim = feature_map.shape
        image_feats = feature_map.reshape(batch_size, h * w, hidden_dim)    
        
        # Prepare text embeddings for classification head
        if text_embeds is not None and input_ids is not None:
            # Project text embeddings: [B*num_queries, projection_dim] -> [B*num_queries, text_hidden_size]
            projected_text_embeds = self.text_projection_for_detection(text_embeds)
            
            num_queries = text_embeds.shape[0]
            # Reshape to [B, num_queries, text_hidden_size]
            query_embeds = projected_text_embeds.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            
            # Create query mask (non-padded queries)
            query_mask = torch.ones(batch_size, num_queries, dtype=torch.bool, device=text_embeds.device)  # First token > 0 means valid query
            
            # Classification predictions
            pred_logits, class_embeds = self.class_head(
                image_feats, query_embeds, query_mask
            )
        else:
            # No text queries
            pred_logits, class_embeds = self.class_head(image_feats)
            query_embeds = None
            
        # Box predictions
        pred_boxes = self.box_head(image_feats)
        
        # Prepare outputs
        outputs = {
            'logits': pred_logits,
            'pred_boxes': pred_boxes,
            'text_embeds': query_embeds,
            'image_embeds': feature_map,
            'class_embeds': class_embeds,
        }
        
        # Add CLIP loss if available
        if 'loss' in clip_outputs:
            outputs['clip_loss'] = clip_outputs['loss']
        
        return outputs

    def generate_text_embeddings(self, text_queries: List[str]) -> torch.Tensor:
        """
        Generate text embeddings สำหรับ inference
        
        Args:
            text_queries: List of Thai text queries
        
        Returns:
            text_embeds: [num_queries, projection_dim]
        """
        text_inputs = self.tokenizer(
            text_queries,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        
        device = next(self.parameters()).device
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            text_outputs = self.backbone.encode_text(input_ids, attention_mask)
            text_embeds = text_outputs['text_embeds']
        
        return text_embeds
    
    def predict_with_text_embeddings(
        self,
        pixel_values: torch.Tensor,
        text_embeds: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict โดยใช้ pre-computed text embeddings
        
        Args:
            pixel_values: [B, 3, H, W]
            text_embeds: [num_queries, projection_dim]
        
        Returns:
            {
                'logits': [B, num_patches, num_queries],
                'pred_boxes': [B, num_patches, 4],
                'class_embeds': [B, num_patches, vision_hidden_size],
                'image_embeds': [B, H, W, vision_hidden_size]
            }
        """
        with torch.no_grad():
            # Get image features
            vision_outputs = self.backbone.encode_image(pixel_values)
            last_hidden_state = vision_outputs['last_hidden_state']
            
            # Process image embeddings (same as forward)
            image_embeds = self.layer_norm(last_hidden_state)
            class_token_out = image_embeds[:, :1, :].expand(-1, image_embeds.shape[1]-1, -1)
            image_embeds = image_embeds[:, 1:, :] * class_token_out
            image_embeds = self.layer_norm(image_embeds)
            
            # Flatten
            batch_size = image_embeds.shape[0]
            image_feats = image_embeds.reshape(batch_size, -1, self.vision_hidden_size)
            
            # Box predictions
            pred_boxes = self.box_head(image_feats)
            
            # Project text embeddings
            projected_text_embeds = self.text_projection_for_detection(text_embeds)
            
            # Expand for batch
            if projected_text_embeds.dim() == 2:
                projected_text_embeds = projected_text_embeds.unsqueeze(0).expand(
                    batch_size, -1, -1
                )
            
            # Classification predictions
            pred_logits, class_embeds = self.class_head(
                image_feats, projected_text_embeds
            )
            
            return {
                'logits': pred_logits,
                'pred_boxes': pred_boxes,
                'class_embeds': class_embeds,
                'image_embeds': image_embeds
            }
            
# =============== TEST CODE =====================
# ==================== Test Code ====================

# if __name__ == '__main__':
#     print("Testing Complete OWL-ViT Model...")
    
#     # Create model
#     model = CustomOWLViTForObjectDetection(
#         d_out=512,
#         freeze_clip=False,
#         num_detection_layers=3
#     )
    
#     # Test inputs
#     batch_size = 2
#     num_queries = 5
    
#     pixel_values = torch.randn(batch_size, 3, 224, 224)
    
#     # Thai queries
#     thai_queries = ["เก้าอี้", "โต๊ะ", "โคมไฟ", "โซฟา", "เตียง"]
#     all_queries = thai_queries * batch_size
    
#     text_inputs = model.tokenizer(
#         all_queries,
#         padding=True,
#         truncation=True,
#         return_tensors='pt'
#     )
    
#     # Forward pass
#     print("\n" + "="*60)
#     print("Testing Forward Pass")
#     print("="*60)
    
#     outputs = model(
#         pixel_values=pixel_values,
#         input_ids=text_inputs['input_ids'],
#         attention_mask=text_inputs['attention_mask']
#     )
    
#     print(f"\nOutput shapes:")
#     print(f"  Logits: {outputs['logits'].shape}")
#     print(f"  Pred boxes: {outputs['pred_boxes'].shape}")
#     print(f"  Image embeds: {outputs['image_embeds'].shape}")
#     print(f"  Class embeds: {outputs['class_embeds'].shape}")
    
#     # Test inference with pre-computed embeddings
#     print("\n" + "="*60)
#     print("Testing Inference with Pre-computed Embeddings")
#     print("="*60)
    
#     text_embeds = model.generate_text_embeddings(thai_queries)
#     print(f"Text embeddings shape: {text_embeds.shape}")
    
#     pred_outputs = model.predict_with_text_embeddings(
#         pixel_values=pixel_values,
#         text_embeds=text_embeds
#     )
    
#     print(f"\nPrediction shapes:")
#     print(f"  Logits: {pred_outputs['logits'].shape}")
#     print(f"  Pred boxes: {pred_outputs['pred_boxes'].shape}")
    
#     print("\n✓ Model test completed successfully!")
    
    """
    ✓ Box Prediction Head initialized
  Vision dim: 768
  Layers: 3
  Output: 4 (normalized coordinates)

================================================================================
Model Summary
================================================================================
Total Parameters: 369,666,821
Trainable Parameters: 369,666,821
Frozen Parameters: 0

Dimensions:
  Vision Hidden Size: 768
  Text Hidden Size: 768
  Projection Dim: 512
  Num Patches: 49 (7x7)
================================================================================


============================================================
Testing Forward Pass
============================================================

Output shapes:
  Logits: torch.Size([2, 49, 5])
  Pred boxes: torch.Size([2, 49, 4])
  Image embeds: torch.Size([2, 7, 7, 768])
  Class embeds: torch.Size([2, 49, 768])

============================================================
Testing Inference with Pre-computed Embeddings
============================================================
Text embeddings shape: torch.Size([5, 512])

Prediction shapes:
  Logits: torch.Size([2, 49, 5])
  Pred boxes: torch.Size([2, 49, 4])

✓ Model test completed successfully!
    """