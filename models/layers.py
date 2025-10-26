import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ====================================================================
# Positional Encoding
# ====================================================================

class PositionalEncoding2D(nn.Module):
    """
    2D Positional Encoding สำหรับ image features
    ใช้กับ feature maps ที่มี spatial dimensions
    """
    
    def __init__(self, d_model: int, max_h: int = 32, max_w: int = 32):
        """
        Args:
            d_model: Feature dimension (ต้องหาร 4 ลงตัว)
            max_h: Maximum height
            max_w: Maximum width
        """
        super().__init__()
        
        assert d_model % 4 == 0, "d_model must be divisible by 4"
        
        self.d_model = d_model
        d_model_half = d_model // 2
        
        # Create positional encoding
        div_term = torch.exp(
            torch.arange(0, d_model_half, 2).float() * 
            (-math.log(10000.0) / d_model_half)
        )
        
        # Height encoding
        pos_h = torch.arange(0, max_h).float().unsqueeze(1)
        pe_h = torch.zeros(max_h, d_model_half)
        pe_h[:, 0::2] = torch.sin(pos_h * div_term)
        pe_h[:, 1::2] = torch.cos(pos_h * div_term)
        
        # Width encoding
        pos_w = torch.arange(0, max_w).float().unsqueeze(1)
        pe_w = torch.zeros(max_w, d_model_half)
        pe_w[:, 0::2] = torch.sin(pos_w * div_term)
        pe_w[:, 1::2] = torch.cos(pos_w * div_term)
        
        # Combine [H, W, D]
        pe = torch.zeros(max_h, max_w, d_model)
        pe[:, :, :d_model_half] = pe_h.unsqueeze(1).repeat(1, max_w, 1)
        pe[:, :, d_model_half:] = pe_w.unsqueeze(0).repeat(max_h, 1, 1)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, D] feature map
        
        Returns:
            x + positional encoding: [B, H, W, D]
        """
        B, H, W, D = x.shape
        return x + self.pe[:H, :W, :].unsqueeze(0)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding (แบบ DETR)
    """
    
    def __init__(self, num_positions: int, d_model: int):
        """
        Args:
            num_positions: Number of positions (e.g., num_patches)
            d_model: Feature dimension
        """
        super().__init__()
        self.position_embeddings = nn.Parameter(
            torch.randn(num_positions, d_model)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] features
        
        Returns:
            x + positional encoding: [B, N, D]
        """
        return x + self.position_embeddings.unsqueeze(0)


# ====================================================================
# Attention Layers
# ====================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [B, N_q, D]
            key: [B, N_k, D]
            value: [B, N_v, D]
            mask: [B, N_q, N_k] (optional)
        
        Returns:
            output: [B, N_q, D]
        """
        B = query.shape[0]
        
        # Linear projections and reshape
        Q = self.q_proj(query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, N_q, d_k]
        K = self.k_proj(key).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)    # [B, H, N_k, d_k]
        V = self.v_proj(value).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, N_v, d_k]
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, N_q, N_k]
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)  # [B, H, N_q, d_k]
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        output = self.out_proj(output)
        
        return output


class CrossAttention(nn.Module):
    """
    Cross-Attention between image and text features
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            image_features: [B, N_img, D] (query)
            text_features: [B, N_text, D] (key, value)
            mask: [B, N_img, N_text] (optional)
        
        Returns:
            output: [B, N_img, D]
        """
        # Cross attention
        attn_output = self.attention(
            query=image_features,
            key=text_features,
            value=text_features,
            mask=mask
        )
        
        # Residual connection + layer norm
        output = self.norm(image_features + self.dropout(attn_output))
        
        return output


# ====================================================================
# Feed-Forward Networks
# ====================================================================

class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network (Transformer-style)
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Args:
            d_model: Input/output dimension
            d_ff: Hidden dimension
            dropout: Dropout rate
            activation: 'relu', 'gelu', or 'silu'
        """
        super().__init__()
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
        
        Returns:
            output: [B, N, D]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MLPBlock(nn.Module):
    """
    MLP Block with residual connection and layer norm
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout, activation)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
        
        Returns:
            output: [B, N, D]
        """
        return self.norm(x + self.dropout(self.ffn(x)))


# ====================================================================
# Feature Fusion Layers
# ====================================================================

class FeatureFusionLayer(nn.Module):
    """
    Feature Fusion Layer สำหรับรวม image และ text features
    """
    
    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        output_dim: int,
        fusion_method: str = 'concat'
    ):
        """
        Args:
            image_dim: Image feature dimension
            text_dim: Text feature dimension
            output_dim: Output dimension
            fusion_method: 'concat', 'add', or 'multiply'
        """
        super().__init__()
        
        self.fusion_method = fusion_method
        
        if fusion_method == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(image_dim + text_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            )
        elif fusion_method == 'add':
            assert image_dim == text_dim, "Dimensions must match for 'add' fusion"
            self.fusion = nn.Sequential(
                nn.Linear(image_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            )
        elif fusion_method == 'multiply':
            assert image_dim == text_dim, "Dimensions must match for 'multiply' fusion"
            self.fusion = nn.Sequential(
                nn.Linear(image_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            image_features: [B, N_img, D_img]
            text_features: [B, N_text, D_text]
        
        Returns:
            fused_features: [B, N, D_out]
        """
        if self.fusion_method == 'concat':
            # Broadcast text to match image spatial dimensions
            B, N_img, D_img = image_features.shape
            _, N_text, D_text = text_features.shape
            
            # Expand text features to match image
            text_expanded = text_features.mean(dim=1, keepdim=True).expand(-1, N_img, -1)
            
            # Concatenate
            combined = torch.cat([image_features, text_expanded], dim=-1)
            fused = self.fusion(combined)
            
        elif self.fusion_method == 'add':
            # Element-wise addition
            text_pooled = text_features.mean(dim=1, keepdim=True).expand_as(image_features)
            combined = image_features + text_pooled
            fused = self.fusion(combined)
            
        elif self.fusion_method == 'multiply':
            # Element-wise multiplication
            text_pooled = text_features.mean(dim=1, keepdim=True).expand_as(image_features)
            combined = image_features * text_pooled
            fused = self.fusion(combined)
        
        return fused


class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive Feature Fusion with learnable gating
    """
    
    def __init__(self, d_model: int):
        """
        Args:
            d_model: Feature dimension
        """
        super().__init__()
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        feature1: torch.Tensor,
        feature2: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            feature1: [B, N, D]
            feature2: [B, N, D]
        
        Returns:
            fused: [B, N, D]
        """
        # Compute gate
        combined = torch.cat([feature1, feature2], dim=-1)
        gate = self.gate(combined)
        
        # Weighted fusion
        fused = gate * feature1 + (1 - gate) * feature2
        
        return self.norm(fused)


# ====================================================================
# Projection Layers
# ====================================================================

class ProjectionHead(nn.Module):
    """
    Projection Head สำหรับ contrastive learning (CLIP-style)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        use_bn: bool = False
    ):
        """
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension (projection space)
            num_layers: Number of layers
            use_bn: Use batch normalization
        """
        super().__init__()
        
        layers = []
        
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            
            layers.append(nn.Linear(in_dim, out_dim))
            
            if i < num_layers - 1:
                if use_bn:
                    layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU(inplace=True))
        
        self.projection = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D_in] or [B, N, D_in]
        
        Returns:
            projected: [B, D_out] or [B, N, D_out]
        """
        return self.projection(x)


# ====================================================================
# Normalization Layers
# ====================================================================

class L2Normalize(nn.Module):
    """
    L2 Normalization layer
    """
    
    def __init__(self, dim: int = -1, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., D]
        
        Returns:
            normalized: [..., D]
        """
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


# ====================================================================
# Utility Layers
# ====================================================================

class Reshape(nn.Module):
    """
    Reshape layer (useful in Sequential)
    """
    
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return x.view(batch_size, *self.shape)


class Permute(nn.Module):
    """
    Permute dimensions (useful in Sequential)
    """
    
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, *self.dims)


# ====================================================================
# Test Code
# ====================================================================

if __name__ == '__main__':
    print("Testing Custom Layers...")
    
    batch_size = 2
    seq_len = 49
    d_model = 768
    
    # Test Positional Encoding
    print("\n" + "="*60)
    print("Testing Positional Encoding 2D")
    print("="*60)
    
    pos_enc = PositionalEncoding2D(d_model, max_h=7, max_w=7)
    x = torch.randn(batch_size, 7, 7, d_model)
    x_pos = pos_enc(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_pos.shape}")
    print(f"✓ Position encoding added")
    
    # Test Multi-Head Attention
    print("\n" + "="*60)
    print("Testing Multi-Head Attention")
    print("="*60)
    
    mha = MultiHeadAttention(d_model, num_heads=8)
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    attn_output = mha(query, key, value)
    print(f"Query shape: {query.shape}")
    print(f"Output shape: {attn_output.shape}")
    print(f"✓ Attention computed")
    
    # Test Cross Attention
    print("\n" + "="*60)
    print("Testing Cross Attention")
    print("="*60)
    
    cross_attn = CrossAttention(d_model, num_heads=8)
    image_feat = torch.randn(batch_size, 49, d_model)
    text_feat = torch.randn(batch_size, 10, d_model)
    
    cross_output = cross_attn(image_feat, text_feat)
    print(f"Image features: {image_feat.shape}")
    print(f"Text features: {text_feat.shape}")
    print(f"Output shape: {cross_output.shape}")
    print(f"✓ Cross attention computed")
    
    # Test Feature Fusion
    print("\n" + "="*60)
    print("Testing Feature Fusion")
    print("="*60)
    
    fusion = FeatureFusionLayer(
        image_dim=768,
        text_dim=768,
        output_dim=512,
        fusion_method='concat'
    )
    
    fused = fusion(image_feat, text_feat)
    print(f"Image features: {image_feat.shape}")
    print(f"Text features: {text_feat.shape}")
    print(f"Fused output: {fused.shape}")
    print(f"✓ Features fused")
    
    # Test Projection Head
    print("\n" + "="*60)
    print("Testing Projection Head")
    print("="*60)
    
    proj_head = ProjectionHead(
        input_dim=768,
        hidden_dim=512,
        output_dim=256,
        num_layers=3
    )
    
    x = torch.randn(batch_size, 768)
    proj_output = proj_head(x)
    print(f"Input shape: {x.shape}")
    print(f"Projected shape: {proj_output.shape}")
    print(f"✓ Projection computed")
    
    # Count parameters
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"\nMultiHeadAttention parameters: {total_params:,}")
    
    print("\n✓ All layer tests passed!")