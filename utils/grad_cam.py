import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2


class GradCAM:
    """
    Grad-CAM for Vision Transformer based object detection
    Specifically designed for CustomOWLViTForObjectDetection
    """
    
    def __init__(self, model):
        """
        Args:
            model: CustomOWLViTForObjectDetection instance
        """
        self.model = model
        self.model.eval()
        
        self.gradients = None
        self.activations = None
        
        # ✅ FIX: Use correct layer path for your architecture
        self.target_layer_name = 'backbone.vision_model.vision_model.encoder'
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        
        target_layer = None
        
        # ✅ Target the LAST encoder layer, not the encoder container
        try:
            # Path: backbone.vision_model.vision_model.encoder.layers[-1]
            if hasattr(self.model, 'backbone'):
                vision_model = self.model.backbone.vision_model.vision_model
                if hasattr(vision_model, 'encoder'):
                    if hasattr(vision_model.encoder, 'layers'):
                        # Get the last encoder layer
                        target_layer = vision_model.encoder.layers[-1]
                        print(f"✓ Found target layer: backbone.vision_model.vision_model.encoder.layers[-1]")
        except Exception as e:
            print(f"⚠️  Error accessing encoder layers: {e}")
        
        if target_layer is None:
            # Fallback: search for last layer
            encoder_layers = []
            for name, module in self.model.named_modules():
                if 'encoder.layers' in name:
                    encoder_layers.append((name, module))
            
            if encoder_layers:
                name, target_layer = encoder_layers[-1]
                print(f"✓ Using fallback: {name}")
        
        if target_layer is None:
            raise ValueError("Could not find vision encoder layer!")
        
        # Forward hook - capture layer output
        def forward_hook(module, input, output):
            # ViT layer output is either tuple or BaseModelOutput
            if isinstance(output, tuple):
                self.activations = output[0]  # (hidden_states, ...)
            elif hasattr(output, 'last_hidden_state'):
                self.activations = output.last_hidden_state
            elif isinstance(output, torch.Tensor):
                self.activations = output
            else:
                # Try to get hidden states
                self.activations = output[0] if isinstance(output, (tuple, list)) else output
            
            # Ensure requires_grad
            if self.activations is not None:
                self.activations = self.activations.requires_grad_(True)
        
        # Backward hook - capture gradients
        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] contains gradients w.r.t. the output
            if grad_output[0] is not None:
                self.gradients = grad_output[0]
        
        # Register hooks
        self.forward_handle = target_layer.register_forward_hook(forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        # Forward hook
        def forward_hook(module, input, output):
            # For CLIPEncoder, output is last_hidden_state
            if isinstance(output, tuple):
                self.activations = output[0]  # BaseModelOutput.last_hidden_state
            elif isinstance(output, torch.Tensor):
                self.activations = output
            else:
                # Handle BaseModelOutput
                if hasattr(output, 'last_hidden_state'):
                    self.activations = output.last_hidden_state
                elif hasattr(output, 'hidden_states'):
                    self.activations = output.hidden_states[-1]
                else:
                    self.activations = output
            
            # Ensure requires_grad
            if self.activations is not None:
                self.activations = self.activations.requires_grad_(True)
        
        # Backward hook
        def backward_hook(module, grad_input, grad_output):
            # Get first non-None gradient
            for grad in grad_output:
                if grad is not None:
                    self.gradients = grad
                    break
        
        # Register
        self.forward_handle = target_layer.register_forward_hook(forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    def remove_hooks(self):
        """Remove hooks"""
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()
    
    def generate_cam(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_patch_idx: Optional[int] = None,
        target_class_idx: Optional[int] = None
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap"""
        
        # Reset
        self.gradients = None
        self.activations = None
        
        # Set to train mode (needed for gradients)
        was_training = self.model.training
        self.model.train()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Get device
        device = next(self.model.parameters()).device
        
        # Move to device
        pixel_values = pixel_values.to(device).requires_grad_(True)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        try:
            # Forward
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs['logits'][0]  # [49, num_classes]
            
            # Select target
            if target_patch_idx is None or target_class_idx is None:
                max_score, max_idx = logits.view(-1).max(0)
                target_patch_idx = max_idx // logits.shape[1]
                target_class_idx = max_idx % logits.shape[1]
            
            # Target score
            target_score = logits[target_patch_idx, target_class_idx]
            
            # Backward
            target_score.backward(retain_graph=False)
            
            # Check if captured
            if self.gradients is None or self.activations is None:
                print("⚠️  Gradients/activations not captured!")
                return np.zeros((7, 7))
            
            # Process gradients and activations
            grads = self.gradients.detach()  # [B, num_patches+1, hidden_dim]
            acts = self.activations.detach()  # [B, num_patches+1, hidden_dim]
            
            # Remove batch dimension
            if grads.dim() == 3:
                grads = grads[0]  # [num_patches+1, hidden_dim]
            if acts.dim() == 3:
                acts = acts[0]  # [num_patches+1, hidden_dim]
            
            # Remove CLS token (position 0)
            if grads.shape[0] == 50:  # 49 patches + 1 CLS
                grads = grads[1:]  # [49, 768]
                acts = acts[1:]   # [49, 768]
            
            # ✅ Method 1: Global average pooling of gradients (weights)
            weights = grads.mean(dim=0, keepdim=True)  # [1, 768]
            
            # Weighted combination
            cam = (weights * acts).sum(dim=1)  # [49]
            
            # ReLU
            cam = F.relu(cam)
            
            # Normalize
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            
            # Reshape to 7x7
            cam = cam[:49].reshape(7, 7)
            cam = cam.cpu().numpy()
            
            return cam
            
        except Exception as e:
            print(f"⚠️  Error in generate_cam: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((7, 7))
        
        finally:
            # Restore mode
            if not was_training:
                self.model.eval()
            self.model.zero_grad()
    
    def visualize_cam(
        self,
        image: Image.Image,
        cam: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Overlay CAM on image"""
        
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # Resize CAM
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Colormap
        cam_colored = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = (alpha * cam_colored + (1 - alpha) * img_array).astype(np.uint8)
        
        return overlay


class GradCAMDetectionVisualizer:
    """Complete Grad-CAM visualization"""
    
    def __init__(
        self,
        model,
        tokenizer,
        class_names: List[str],
        thai_class_names: List[str],
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.class_names = class_names
        self.thai_class_names = thai_class_names
        
        # Create Grad-CAM
        self.grad_cam = GradCAM(self.model)
    
    def preprocess_image(self, image_path: str, image_size: int = 224):
        """Load and preprocess"""
        
        image = Image.open(image_path).convert('RGB')
        orig_size = image.size
        
        image_resized = image.resize((image_size, image_size))
        
        img_array = np.array(image_resized).astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor, image, orig_size
    
    @torch.no_grad()
    def get_predictions(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        confidence_threshold: float = 0.3
    ):
        """Get predictions"""
        
        outputs = self.model(pixel_values, input_ids, attention_mask)
        
        logits = outputs['logits'][0]
        boxes = outputs['pred_boxes'][0]
        
        scores, labels = logits.max(dim=-1)
        
        mask = scores > confidence_threshold
        
        return {
            'boxes': boxes[mask],
            'labels': labels[mask],
            'scores': scores[mask],
            'patch_indices': torch.where(mask)[0]
        }
    
    def visualize_detection_with_gradcam(
        self,
        image_path: str,
        confidence_threshold: float = 0.3,
        save_path: Optional[str] = None
    ):
        """Visualize detections with Grad-CAM"""
        
        print("="*80)
        print("🎨 Grad-CAM Detection Visualization")
        print("="*80)
        print(f"Image: {image_path}")
        print(f"Threshold: {confidence_threshold}")
        print("="*80 + "\n")
        
        # Preprocess
        img_tensor, orig_image, orig_size = self.preprocess_image(image_path)
        img_tensor = img_tensor.to(self.device)
        
        # Tokenize
        text_inputs = self.tokenizer(
            self.thai_class_names,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors='pt'
        )
        input_ids = text_inputs['input_ids'].to(self.device)
        attention_mask = text_inputs['attention_mask'].to(self.device)
        
        # Get predictions
        predictions = self.get_predictions(
            img_tensor, input_ids, attention_mask, confidence_threshold
        )
        
        num_detections = len(predictions['scores'])
        print(f"✓ Found {num_detections} detections\n")
        
        if num_detections == 0:
            print("⚠️  No detections!")
            return
        
        # Visualize
        num_cols = min(3, num_detections)
        num_rows = (num_detections + num_cols - 1) // num_cols + 1
        
        fig = plt.figure(figsize=(6*num_cols, 6*num_rows))
        
        # Original with boxes
        ax = plt.subplot(num_rows, num_cols, 1)
        ax.imshow(orig_image)
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.class_names)))
        
        for box, label, score in zip(
            predictions['boxes'],
            predictions['labels'],
            predictions['scores']
        ):
            x_c, y_c, w, h = box.cpu().numpy()
            
            orig_w, orig_h = orig_size
            x_c_px = x_c * orig_w
            y_c_px = y_c * orig_h
            w_px = w * orig_w
            h_px = h * orig_h
            
            x_min = x_c_px - w_px / 2
            y_min = y_c_px - h_px / 2
            
            color = colors[label.item()]
            
            rect = patches.Rectangle(
                (x_min, y_min), w_px, h_px,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            ax.text(
                x_min, y_min - 5,
                f"{self.thai_class_names[label.item()]}\n{score:.2f}",
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                fontsize=9, color='white', fontweight='bold'
            )
        
        ax.set_title('All Detections', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Grad-CAM for each detection
        for idx, (patch_idx, class_idx, score) in enumerate(zip(
            predictions['patch_indices'],
            predictions['labels'],
            predictions['scores']
        )):
            ax = plt.subplot(num_rows, num_cols, idx + 2)
            
            print(f"Generating Grad-CAM {idx+1}/{num_detections}...")
            
            # Generate CAM
            cam = self.grad_cam.generate_cam(
                pixel_values=img_tensor,
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_patch_idx=patch_idx.item(),
                target_class_idx=class_idx.item()
            )
            
            # Overlay
            overlay = self.grad_cam.visualize_cam(orig_image, cam, alpha=0.5)
            
            ax.imshow(overlay)
            
            # Draw box
            box = predictions['boxes'][idx]
            x_c, y_c, w, h = box.cpu().numpy()
            
            orig_w, orig_h = orig_size
            x_c_px = x_c * orig_w
            y_c_px = y_c * orig_h
            w_px = w * orig_w
            h_px = h * orig_h
            
            x_min = x_c_px - w_px / 2
            y_min = y_c_px - h_px / 2
            
            color = colors[class_idx.item()]
            
            rect = patches.Rectangle(
                (x_min, y_min), w_px, h_px,
                linewidth=3, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            ax.set_title(
                f"{self.thai_class_names[class_idx.item()]}\n{score:.3f}",
                fontsize=10, fontweight='bold'
            )
            ax.axis('off')
        
        plt.suptitle(
            f'Grad-CAM - {Path(image_path).name}',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Saved: {save_path}")
        
        plt.show()
        
        print("\n" + "="*80)
        print("✓ Complete!")
        print("="*80)


if __name__ == "__main__":
    print("Grad-CAM module loaded!")