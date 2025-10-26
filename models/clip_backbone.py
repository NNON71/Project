import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModel, AutoTokenizer, ResNetModel
from typing import Dict, Optional

class OWLViTCLIPBackbone(nn.Module) :
    """
    CLIP Backbone using OWL-ViT Vision Model.
    - Vision Encoder : "clip/clip-vit-base-patch32"
    - Text Encoder : "clicknext/phayathaibert"
    """
    
    def __init__(
        self,
        d_out: int = 512,
        image_encoder_name: str = "openai/clip-vit-base-patch32",
        text_encoder_name: str = "clicknext/phayathaibert",
        freeze_vision: bool = False,
        freeze_text: bool = False
    ) :
        """ 

        Args:
            d_out (int, optional): Output dimension. Defaults to 512.
            image_encoder_name (str, optional): Image encoder model name. Defaults to "openai/clip-vit-base-patch32".
            text_encoder_name (str, optional): Text encoder model name. Defaults to "clicknext/phayathaibert".
            freeze_vision (bool, optional): Whether to freeze vision encoder. Defaults to True.
            freeze_text (bool, optional): Whether to freeze text encoder. Defaults to True.
        """
        super().__init__()
        
        # Vision Encoder
        self.vision_model = CLIPVisionModel.from_pretrained(image_encoder_name)
        vision_hidden_size = self.vision_model.config.hidden_size
        
        # Text Encoder
        self.text_model = AutoModel.from_pretrained(text_encoder_name)
        text_hidden_size = self.text_model.config.hidden_size
        
        # Tokenizer 
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        
        # Projection layers
        self.vision_projection = nn.Linear(d_out, d_out, bias=False)
        self.text_projection = nn.Linear(d_out, d_out, bias=False)
        
        self.vision_pre_projection = nn.Linear(vision_hidden_size, d_out, bias=False)  # 768→512
        self.text_pre_projection = nn.Linear(text_hidden_size, d_out, bias=False)      # 768→512
        
        # logit scale parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        
        # Freeze parameters if specified
        if freeze_vision :
            for param in self.vision_model.parameters() :
                param.requires_grad = False

        if freeze_text :
            for param in self.text_model.parameters() :
                param.requires_grad = False
                
    def encode_image(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor] :
        """Encode images to feature vectors.

        Args:
            pixel_values (torch.Tensor): Input images tensor of shape (B, C, H, W).

        Returns:
            {
                'last_hidden_state': [B, num_patches+1, vision_hidden_size], # Patch embeddings
                'pooler_output': [B, vision_hidden_size], # CLS token representation
                'image_embeds': [B, d_out] # projected image features
            }
        """
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        
        # Get pooler output (CLS token)
        pooler_output = vision_outputs.pooler_output  # [B, vision_hidden_size]
        pre_embeds = self.vision_pre_projection(pooler_output)  # [B, 768] → [B, 512]
        image_embeds = self.vision_projection(pre_embeds)       # [B, 512] → [B, 512]
        
        # Project to d_out
        # image_embeds = self.vision_projection(pooler_output)  # [B, d_out]
        
        return {
            'last_hidden_state': vision_outputs.last_hidden_state,
            'pooler_output': pooler_output,
            'image_embeds': image_embeds
        }
        
    def encode_text(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor] :
        """Encode text to feature vectors.

        Args:
            input_ids (torch.Tensor): Input token ids tensor of shape (B, seq_len).
            attention_mask (Optional[torch.Tensor], optional): Attention mask tensor of shape (B, seq_len). Defaults to None.

        Returns:
            {
                'last_hidden_state': [B, seq_len, text_hidden_size], # Token embeddings
                'pooler_output': [B, text_hidden_size], # CLS token representation
                'text_embeds': [B, d_out] # projected text features
            }
        """
        text_outputs = self.text_model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        # Get pooler output (CLS token)
        if hasattr(text_outputs, 'pooler_output'):
            pooler_output = text_outputs.pooler_output  # [B, 768]
        else:
            pooler_output = text_outputs.last_hidden_state[:, 0, :]  # [B, 768]
        
        # Project to d_out
        # text_embeds = self.text_projection(pooler_output)  # [B, d_out]
        
        pre_embeds = self.text_pre_projection(pooler_output)  # 768→512
        text_embeds = self.text_projection(pre_embeds)        # 512→512
        
        return {
            'last_hidden_state': text_outputs.last_hidden_state,
            'pooler_output': pooler_output,
            'text_embeds': text_embeds
        }
        
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = False
    ) -> Dict[str, torch.Tensor] :
        """Forward pass for CLIP Backbone.

        Args:
            pixel_values (torch.Tensor): Input images tensor of shape (B, C, H, W).
            input_ids (Optional[torch.Tensor], optional): Input token ids tensor of shape (B, seq_len). Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Attention mask tensor of shape (B, seq_len). Defaults to None.
            return_loss (bool, optional): Whether to compute contrastive loss. Defaults to False.
        Returns:
            {
                'image_embeds': [B, d_out],
                'text_embeds': [B, d_out],
                'vision_model_outputs': Dict,
                'text_model_outputs': Dict,
                'logit_scale': scalar,
                'loss': scalar (if return_loss is True)
            }
        """
        
        # Encode images
        vision_outputs = self.encode_image(pixel_values)
        image_embeds = vision_outputs['image_embeds']
        
        outputs = {
            'image_embeds': image_embeds,
            'vision_model_output': vision_outputs,
            'logit_scale': self.logit_scale.exp()
        }
        
        # Encoder text if provided
        if input_ids is not None :
            text_outputs = self.encode_text(input_ids, attention_mask)
            text_embeds = text_outputs['text_embeds']
            
            outputs.update({
                'text_embeds': text_embeds,
                'text_model_outputs': text_outputs
            })
            
            # Compute contrastive loss if specified
            if return_loss :
                loss = self.compute_contrastive_loss(image_embeds, text_embeds)
                outputs['loss'] = loss
            
        return outputs
    
    def compute_contrastive_loss(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor
    ) -> torch.Tensor :
        """Compute contrastive loss between image and text embeddings.

        Args:
            image_embeds (torch.Tensor): Image embeddings of shape (B, d_out).
            text_embeds (torch.Tensor): Text embeddings of shape (B, d_out).
        Returns:
            torch.Tensor: Contrastive loss scalar.
        """
        
        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # Compute logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()
        
        # Labels
        batch_size = image_embeds.size(0)
        labels = torch.arange(batch_size).to(image_embeds.device)
        
        # Contrastive loss
        loss_i2t = nn.functional.cross_entropy(logits_per_image, labels)
        loss_t2i = nn.functional.cross_entropy(logits_per_text, labels)
        
        # Average loss
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss
    

## ======================= Testing ======================== ##

# print("Testing CLIP Backbone...")

# model = OWLViTCLIPBackbone(
#     d_out=512,
#     freeze_vision=False,
#     freeze_text=False
# )

# batch_size = 2
# pixel_values = torch.randn(batch_size, 3, 224, 224)

# text = ["เก้าอี้ในห้องทำงาน", "โต๊ะไม้ในห้องครัว"]
# text_inputs = model.tokenizer(
#     text,
#     padding=True,
#     truncation=True,
#     return_tensors="pt"
# )

# # Forward pass
# outputs = model(
#     pixel_values=pixel_values,
#     input_ids=text_inputs['input_ids'],
#     attention_mask=text_inputs['attention_mask'],
#     return_loss=True
# )   

# print(f"\nOutput keys: {outputs.keys()}")
# print(f"Image embeds shape: {outputs['image_embeds'].shape}")
# print(f"Text embeds shape: {outputs['text_embeds'].shape}")
# print(f"Loss: {outputs['loss'].item():.4f}")
# print(f"Logit scale: {outputs['logit_scale'].item():.2f}")

# # Count parameters
# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f"\nParameters:")
# print(f"  Total: {total_params:,}")
# print(f"  Trainable: {trainable_params:,}")

"""
# OUTPUT 
Output keys: dict_keys(['image_embeds', 'vision_model_output', 'logit_scale', 'text_embeds', 'text_model_outputs', 'loss'])
Image embeds shape: torch.Size([2, 512])
Text embeds shape: torch.Size([2, 512])
Loss: 0.7081
Logit scale: 14.29

Parameters:
  Total: 365,716,225
  Trainable: 365,716,225
"""