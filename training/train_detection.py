import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import wandb
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

from models.owlvit_model import CustomOWLViTForObjectDetection
from mydatasets.detection_dataset import CustomObjectDetectionDataset, detection_collate_fn
from utils.metrics import compute_map
from utils.losses import compute_giou_loss

class OWLViTDetectionTrainer :
    """
    Trainer class for OWL-ViT object detection model
    """
    
    def __init__(self, config: Dict, clip_checkpoint_path: str = None) :
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # set seed
        self._set_seed(config['experiment']['seed'])
        
        # Initialize WandB
        if config['wandb']['enabled'] :
            wandb.init(
                project=config['wandb']['project'],
                entity=config['wandb'].get('entity', None),
                name=config['wandb']['run_name'],
                config=config,
                tags=['clip', 'stage1']
            )
            
        # Build Model
        self.model = self._build_model(clip_checkpoint_path=clip_checkpoint_path)
        
        # Setup Dataset
        self.train_loader, self.val_loader = self._setup_datasets()
        
        # Setup Optimizer and Scheduler
        self.optimizer, self.scheduler = self._setup_optimizer()
        
        # Loss functions
        self.setup_losses()
        
        # Mixed precision
        self.use_amp = config['training']['mixed_precision']
        self.scaler = GradScaler(device=self.device) if self.use_amp else None
        
        # Training state
        self.start_epoch = 0
        self.best_map = 0.0
        self.global_step = 0
        
        # Checkpoint dir
        self.checkpoint_dir = Path(config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def _set_seed(self, seed: int):
        """Set random seed"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        print(f"✓ Random seed set to {seed}")
    
    def _build_model(self, clip_checkpoint_path) -> nn.Module :
        """Build OWL-ViT detection model"""
        
        model = CustomOWLViTForObjectDetection(
            d_out=self.config['model']['projection_dim'],
            image_encoder_name=self.config['model']['image_encoder'],
            text_encoder_name=self.config['model']['text_encoder'],
            freeze_clip=self.config['model']['freeze_clip_backbone']
        )
        
        # load pretrained CLIP Backbone
        if clip_checkpoint_path or self.config['model']['clip_checkpoint_path']:
            checkpoint_path = clip_checkpoint_path or self.config['model']['clip_checkpoint_path']
            print(f"\nLoading pretrained CLIP from: {checkpoint_path}")
            
            clip_checkpoint = torch.load(checkpoint_path, map_location=self.device)
            clip_state_dict = clip_checkpoint.get('model_state_dict', clip_checkpoint)
            
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            
            # Map extracted CLIP keys to model keys
            for clip_key, clip_value in clip_state_dict.items():
                model_key = None
                # After extraction, keys should be clean:
                # vision_model.X → backbone.vision_model.vision_model.X
                # text_model.X → backbone.text_model.text_model.X
                
                if clip_key.startswith('vision_model.'):
                    # May need to add double prefix
                    model_key = f"backbone.vision_model.{clip_key}"
                elif clip_key.startswith('text_model.'):
                    model_key = f"backbone.text_model.{clip_key}"
                    
                elif clip_key == 'visual_projection.weight':
                    # Handle visual projection specifically
                    possible_keys = [
                        "backbone.vision_projection.weight",
                        "backbone.visual_projection.weight",
                    ]
                    for test_key in possible_keys:
                        if test_key in model_state_dict and clip_value.shape == model_state_dict[test_key].shape:
                            filtered_state_dict[test_key] = clip_value
                            print(f'✓ visual_projection mapped: {clip_key} -> {test_key}')
                            break
                    continue  # Skip the rest of the logic for this key
                
                elif clip_key == 'text_projection.weight':
                    # Handle text projection specifically
                    test_key = "backbone.text_projection.weight"
                    if test_key in model_state_dict and clip_value.shape == model_state_dict[test_key].shape:
                        filtered_state_dict[test_key] = clip_value
                        print(f'✓ text_projection mapped: {clip_key} -> {test_key}')
                        continue  # Skip the rest of the logic for this key
                    
                elif 'projection' in clip_key:
                        # Handle other projection layers
                        if 'vision' in clip_key:
                            model_key = f"backbone.{clip_key}"
                        elif 'text' in clip_key:
                            model_key = f"backbone.{clip_key}"
                        else:
                            model_key = f"backbone.{clip_key}"
                else:
                    # logit_scale etc
                    model_key = f"backbone.{clip_key}"
                
                if model_key is None:
                    continue
                
                # Try to match
                if model_key in model_state_dict:
                    if clip_value.shape == model_state_dict[model_key].shape:
                        filtered_state_dict[model_key] = clip_value
                        
                if model_key in model_state_dict:
                    if clip_value.shape == model_state_dict[model_key].shape:
                        filtered_state_dict[model_key] = clip_value
                        
                else:
                    # Try alternative: remove one level of nesting
                    if 'vision_model.vision_model.' in model_key:
                        alt_key = model_key.replace('vision_model.vision_model.', 'vision_model.')
                        if alt_key in model_state_dict and clip_value.shape == model_state_dict[alt_key].shape:
                            filtered_state_dict[alt_key] = clip_value
                            continue
                    
                    if 'text_model.text_model.' in model_key:
                        alt_key = model_key.replace('text_model.text_model.', 'text_model.')
                        if alt_key in model_state_dict and clip_value.shape == model_state_dict[alt_key].shape:
                            filtered_state_dict[alt_key] = clip_value
                            continue
            
            # Load
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            
            print(f"✓ Loaded {len(filtered_state_dict)} parameters from CLIP checkpoint")
            
            # Breakdown
            vision_cnt = len([k for k in filtered_state_dict if 'vision' in k])
            text_cnt = len([k for k in filtered_state_dict if 'text' in k])
            proj_cnt = len([k for k in filtered_state_dict if 'projection' in k])
            other_cnt = len(filtered_state_dict) - vision_cnt - text_cnt - proj_cnt
            
            print(f"  Components loaded:")
            print(f"    - Vision: {vision_cnt}")
            print(f"    - Text: {text_cnt}")
            print(f"    - Projections: {proj_cnt}")
            print(f"    - Other: {other_cnt}")
            
            detection_missing = [k for k in missing_keys if any(x in k for x in 
                ['class_head', 'box_head', 'layer_norm', 'text_projection_for_detection'])]
            print(f"  Missing (detection heads): {len(detection_missing)} [Expected]")
        
        model = model.to(self.device)
        
        # Print trainable components
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel Statistics:")
        print(f"  Total params: {total:,}")
        print(f"  Trainable params: {trainable:,}")
        print(f"  Frozen params: {total - trainable:,}")
        
        print(f"\nTrainable Components:")
        components = set()
        for name, param in model.named_parameters():
            if param.requires_grad:
                component = name.split('.')[0]
                components.add(component)
        for comp in sorted(components):
            print(f"  ✓ {comp}")
        
        return model
    
    def _setup_datasets(self):
        """Setup detection datasets"""
        print("\nSetting up detection datasets...")
        
        train_dataset = CustomObjectDetectionDataset(
            annotations_file=self.config['dataset']['train_annotations'],
            images_dir=self.config['dataset']['train_images'],
            class_names=self.config['dataset']['class_names'],
            thai_class_names=self.config['dataset']['thai_class_names'],
            image_size=self.config['dataset']['image_size'],
            max_objects=self.config['dataset']['max_objects'],
            augment=True
        )
        
        val_dataset = CustomObjectDetectionDataset(
            annotations_file=self.config['dataset']['val_annotations'],
            images_dir=self.config['dataset']['val_images'],
            class_names=self.config['dataset']['class_names'],
            thai_class_names=self.config['dataset']['thai_class_names'],
            image_size=self.config['dataset']['image_size'],
            max_objects=self.config['dataset']['max_objects'],
            augment=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['dataset']['num_workers'],
            collate_fn=detection_collate_fn,
            pin_memory=False,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['evaluation']['batch_size'],
            shuffle=False,
            num_workers=self.config['dataset']['num_workers'],
            collate_fn=detection_collate_fn,
            pin_memory=False
        )
        
        print(f"✓ Train: {len(train_dataset)} images, {len(train_loader)} batches")
        print(f"✓ Val: {len(val_dataset)} images, {len(val_loader)} batches")
        print(f"✓ Classes: {len(self.config['dataset']['class_names'])}")
        
        return train_loader, val_loader
    
    def _setup_optimizer(self):
        """Setup optimizer with different LR for backbone and heads"""
        param_groups = []
        
        # CLIP backbone (if trainable)
        clip_params = []
        detection_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'clip.' in name:
                clip_params.append(param)
            else:
                detection_params.append(param)
        
        # Add parameter groups
        if len(clip_params) > 0:
            param_groups.append({
                'params': clip_params,
                'lr': self.config['training']['learning_rate'] * \
                     self.config['training'].get('backbone_lr_mult', 0.1),
                'name': 'clip_backbone'
            })
            print(f"✓ CLIP backbone: {len(clip_params)} param tensors (LR multiplier: {self.config['training'].get('backbone_lr_mult', 0.1)})")
        
        if len(detection_params) > 0:
            param_groups.append({
                'params': detection_params,
                'lr': self.config['training']['learning_rate'],
                'name': 'detection_heads'
            })
            print(f"✓ Detection heads: {len(detection_params)} param tensors (Full LR)")
        
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config['training']['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config['training']['scheduler_t0'],
            T_mult=1,
            eta_min=self.config['training']['min_lr']
        )
        
        print(f"\n✓ Optimizer: AdamW")
        print(f"✓ Scheduler: CosineAnnealingWarmRestarts (T_0={self.config['training']['scheduler_t0']})")
        
        return optimizer, scheduler
    
    def setup_losses(self):
        """Setup loss functions"""
        self.class_loss_fn = nn.CrossEntropyLoss()
        self.box_loss_fn = nn.L1Loss()
        
        # Loss weights
        self.class_loss_weight = self.config['training']['class_loss_weight']
        self.box_loss_weight = self.config['training']['box_loss_weight']
        self.giou_loss_weight = self.config['training']['giou_loss_weight']
        self.clip_loss_weight = self.config['training'].get('clip_loss_weight', 0.0)
        
    def match_predictions_to_targets(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        target_labels: torch.Tensor,
        target_boxes: torch.Tensor
    ):
        """
        Simple Hungarian-style matching
        
        Args:
            pred_logits: [B, num_patches, num_classes]
            pred_boxes: [B, num_patches, 4]
            target_labels: [B, max_objects]
            target_boxes: [B, max_objects, 4]
        
        Returns:
            Matched predictions and targets
        """
        batch_size = pred_logits.shape[0]
        
        all_matched_pred_logits = []
        all_matched_pred_boxes = []
        all_matched_target_labels = []
        all_matched_target_boxes = []
        
        for b in range(batch_size):
            # Get valid targets (label >= 0)
            valid_mask = target_labels[b] >= 0
            if valid_mask.sum() == 0:
                continue
            
            valid_target_labels = target_labels[b][valid_mask]
            valid_target_boxes = target_boxes[b][valid_mask]
            
            num_targets = valid_mask.sum().item()
            num_queries = pred_logits.shape[1]
            
            # Select top-k predictions by confidence
            pred_scores, _ = pred_logits[b].max(dim=-1)
            topk = min(num_targets, num_queries)
            _, topk_indices = torch.topk(pred_scores, topk)
            
            matched_pred_logits = pred_logits[b][topk_indices]
            matched_pred_boxes = pred_boxes[b][topk_indices]
            
            # Match with targets
            if topk > num_targets:
                # Repeat targets
                repeat_factor = (topk // num_targets) + 1
                matched_target_labels = valid_target_labels.repeat(repeat_factor)[:topk]
                matched_target_boxes = valid_target_boxes.repeat(repeat_factor, 1)[:topk]
            else:
                matched_target_labels = valid_target_labels[:topk]
                matched_target_boxes = valid_target_boxes[:topk]
            
            all_matched_pred_logits.append(matched_pred_logits)
            all_matched_pred_boxes.append(matched_pred_boxes)
            all_matched_target_labels.append(matched_target_labels)
            all_matched_target_boxes.append(matched_target_boxes)
        
        if len(all_matched_pred_logits) == 0:
            return None, None, None, None
        
        return (
            torch.cat(all_matched_pred_logits, dim=0),
            torch.cat(all_matched_pred_boxes, dim=0),
            torch.cat(all_matched_target_labels, dim=0),
            torch.cat(all_matched_target_boxes, dim=0)
        )
        
    def compute_detection_loss(
        self,
        outputs: Dict,
        targets: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detection losses
        
        Returns:
            {
                'total_loss': scalar,
                'class_loss': scalar,
                'box_loss': scalar,
                'giou_loss': scalar,
                'clip_loss': scalar
            }
        """
        losses = {}
        
        pred_logits = outputs['logits']
        pred_boxes = outputs['pred_boxes']
        target_labels = targets['labels']
        target_boxes = targets['boxes']
        
        # Match predictions to targets
        matched_pred_logits, matched_pred_boxes, matched_target_labels, matched_target_boxes = \
            self.match_predictions_to_targets(pred_logits, pred_boxes, target_labels, target_boxes)
        
        # Classification loss
        if matched_pred_logits is not None:
            losses['class_loss'] = self.class_loss_fn(matched_pred_logits, matched_target_labels)
        else:
            losses['class_loss'] = torch.tensor(0.0, device=self.device)
        
        # Box losses
        if matched_pred_boxes is not None and matched_target_boxes is not None:
            # L1 loss
            losses['box_loss'] = self.box_loss_fn(matched_pred_boxes, matched_target_boxes)
            
            # GIoU loss
            try:
                losses['giou_loss'] = compute_giou_loss(matched_pred_boxes, matched_target_boxes)
            except Exception as e:
                losses['giou_loss'] = torch.tensor(0.0, device=self.device)
        else:
            losses['box_loss'] = torch.tensor(0.0, device=self.device)
            losses['giou_loss'] = torch.tensor(0.0, device=self.device)
        
        # CLIP loss (optional)
        if 'clip_loss' in outputs and self.clip_loss_weight > 0:
            losses['clip_loss'] = outputs['clip_loss']
        else:
            losses['clip_loss'] = torch.tensor(0.0, device=self.device)
        
        # Total weighted loss
        losses['total_loss'] = (
            self.class_loss_weight * losses['class_loss'] +
            self.box_loss_weight * losses['box_loss'] +
            self.giou_loss_weight * losses['giou_loss'] +
            self.clip_loss_weight * losses['clip_loss']
        )
        
        return losses
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        
        # Set CLIP to eval if frozen
        if self.config['model']['freeze_clip_backbone']:
            self.model.backbone.eval()
        
        metrics = {
            'total_loss': 0.0,
            'class_loss': 0.0,
            'box_loss': 0.0,
            'giou_loss': 0.0,
            'clip_loss': 0.0
        }
        
        pbar = tqdm(
            self.train_loader,
            desc=f"[Stage 2 - Detection] Epoch {epoch+1}/{self.config['training']['epochs']}"
        )
        
        for batch_idx, batch in enumerate(pbar):
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            targets = {
                'labels': batch['labels'].to(self.device),
                'boxes': batch['boxes'].to(self.device)
            }
            
            # Forward
            if self.use_amp:
                with autocast(device_type=self.device.type, dtype=torch.float16):
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    losses = self.compute_detection_loss(outputs, targets)
                
                # Backward
                self.optimizer.zero_grad()
                self.scaler.scale(losses['total_loss']).backward()
                
                if self.config['training']['grad_clip_norm'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip_norm']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                losses = self.compute_detection_loss(outputs, targets)
                
                self.optimizer.zero_grad()
                losses['total_loss'].backward()
                
                if self.config['training']['grad_clip_norm'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip_norm']
                    )
                
                self.optimizer.step()
            
            self.scheduler.step(epoch + batch_idx / len(self.train_loader))
            
            # Update metrics
            for key in metrics.keys():
                metrics[key] += losses[key].item()
            
            # Progress bar
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'cls': f"{losses['class_loss'].item():.3f}",
                'box': f"{losses['box_loss'].item():.3f}",
                'giou': f"{losses['giou_loss'].item():.3f}",
                'lr': f"{self.optimizer.param_groups[-1]['lr']:.6f}"
            })
            
            # Log to W&B
            if self.config['wandb']['enabled'] and \
               batch_idx % self.config['wandb']['log_interval'] == 0:
                log_dict = {
                    'det_train/total_loss': losses['total_loss'].item(),
                    'det_train/class_loss': losses['class_loss'].item(),
                    'det_train/box_loss': losses['box_loss'].item(),
                    'det_train/giou_loss': losses['giou_loss'].item(),
                    'det_train/clip_loss': losses['clip_loss'].item(),
                    'det_train/lr': self.optimizer.param_groups[-1]['lr'],
                    'epoch': epoch,
                    'step': self.global_step
                }
                
                if len(self.optimizer.param_groups) > 1:
                    log_dict['det_train/backbone_lr'] = self.optimizer.param_groups[0]['lr']
                
                wandb.log(log_dict)
            
            self.global_step += 1
        
        # Average metrics
        for key in metrics.keys():
            metrics[key] /= len(self.train_loader)
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[Dict[str, float], float]:
        """Validate model and compute mAP"""
        self.model.eval()
        
        metrics = {
            'total_loss': 0.0,
            'class_loss': 0.0,
            'box_loss': 0.0,
            'giou_loss': 0.0
        }
        
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.val_loader, desc="[Detection] Validation")
        
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            targets = {
                'labels': batch['labels'].to(self.device),
                'boxes': batch['boxes'].to(self.device)
            }
            
            # Forward pass
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Compute losses
            losses = self.compute_detection_loss(outputs, targets)
            
            for key in metrics.keys():
                metrics[key] += losses[key].item()
            
            # Extract predictions
            pred_logits = outputs['logits']  # [B, num_patches, num_classes]
            pred_boxes = outputs['pred_boxes']  # [B, num_patches, 4]
            
            batch_size = pred_logits.shape[0]
            
            # Process each image in batch
            for i in range(batch_size):
                # Get scores and labels
                scores, labels = pred_logits[i].max(dim=-1)  # [num_patches]
                
                # Apply confidence threshold
                conf_threshold = self.config['evaluation']['confidence_threshold']
                conf_mask = scores > conf_threshold
                
                # Append predictions
                all_predictions.append({
                    'boxes': pred_boxes[i][conf_mask].cpu(),
                    'labels': labels[conf_mask].cpu(),
                    'scores': scores[conf_mask].cpu()
                })
                
                # Append ground truth (filter padding)
                valid_mask = targets['labels'][i] >= 0
                all_targets.append({
                    'boxes': targets['boxes'][i][valid_mask].cpu(),
                    'labels': targets['labels'][i][valid_mask].cpu()
                })
        
        # Average losses
        for key in metrics.keys():
            metrics[key] /= len(self.val_loader)
        
        # Compute mAP
        try:
            map_score = compute_map(
                all_predictions,
                all_targets,
                num_classes=len(self.config['dataset']['class_names']),
                iou_threshold=self.config['evaluation']['iou_threshold']
            )
            print(f"\n  ✓ mAP@{self.config['evaluation']['iou_threshold']}: {map_score:.4f}")
        except Exception as e:
            import traceback
            print(f"\n  ✗ Error computing mAP:")
            print(f"    {str(e)}")
            print(f"    Traceback:")
            print(traceback.format_exc())
            map_score = 0.0
        
        # Log to W&B
        if self.config['wandb']['enabled']:
            wandb.log({
                'det_val/total_loss': metrics['total_loss'],
                'det_val/class_loss': metrics['class_loss'],
                'det_val/box_loss': metrics['box_loss'],
                'det_val/giou_loss': metrics['giou_loss'],
                'det_val/mAP': map_score,
                'epoch': epoch
            })
        
        return metrics, map_score
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_map': self.best_map,
            'config': self.config
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save last
        if self.config['checkpoint']['save_last']:
            last_path = self.checkpoint_dir / 'last.pth'
            torch.save(checkpoint, last_path)
        
        # Save best
        if is_best and self.config['checkpoint']['save_best']:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best checkpoint (mAP: {self.best_map:.4f})")
        
        # Save epoch
        if (epoch + 1) % self.config['training']['save_interval'] == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{epoch+1}.pth'
            torch.save(checkpoint, epoch_path)
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("Starting OWL-ViT Detection Training (Stage 2)")
        print("="*80 + "\n")
        
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
            print(f"  Train - Loss: {train_metrics['total_loss']:.4f}")
            print(f"    Class: {train_metrics['class_loss']:.4f}, Box: {train_metrics['box_loss']:.4f}, GIoU: {train_metrics['giou_loss']:.4f}")
            
            # Validate
            val_metrics, map_score = self.validate(epoch)
            
            print(f"  Val - Loss: {val_metrics['total_loss']:.4f}, mAP: {map_score:.4f}")
            
            # Save checkpoint
            is_best = map_score > self.best_map
            if is_best:
                self.best_map = map_score
            
            self.save_checkpoint(epoch, is_best)
        
        print(f"\n{'='*80}")
        print(f"Training Completed! Best mAP: {self.best_map:.4f}")
        print(f"{'='*80}\n")
        
        if self.config['wandb']['enabled']:
            wandb.finish()
            
def main():
    parser = argparse.ArgumentParser(description='Train OWL-ViT Detection (Stage 2)')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--clip-checkpoint', type=str, required=True, help='Path to CLIP checkpoint')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--resume-weights-only', action='store_true', 
                       help='Resume model weights only (skip optimizer/scheduler)')  # ✅ NEW
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = OWLViTDetectionTrainer(config, args.clip_checkpoint)
    
    # Resume if specified
    if args.resume:
        print("\n" + "="*80)
        print("Resuming Training from Checkpoint")
        print("="*80)
        print(f"Loading: {args.resume}")
        
        if not Path(args.resume).exists():
            print(f"❌ Error: Checkpoint not found at {args.resume}")
            return
        
        try:
            checkpoint = torch.load(args.resume, map_location=trainer.device)
            
            # ✅ Always load model state
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Loaded model weights")
            
            # ✅ Conditionally load optimizer/scheduler
            if args.resume_weights_only:
                print("⚠️  Skipping optimizer and scheduler (weights only mode)")
                trainer.start_epoch = 0
                trainer.best_map = 0.0
                trainer.global_step = 0
            else:
                # Try to load optimizer state
                try:
                    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("✓ Loaded optimizer state")
                except (ValueError, KeyError) as e:
                    print(f"⚠️  Could not load optimizer state: {e}")
                    print("   Continuing with fresh optimizer...")
                
                # Try to load scheduler state
                try:
                    trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("✓ Loaded scheduler state")
                except (ValueError, KeyError) as e:
                    print(f"⚠️  Could not load scheduler state: {e}")
                    print("   Continuing with fresh scheduler...")
                
                # Load training state
                trainer.start_epoch = checkpoint.get('epoch', 0) + 1
                trainer.best_map = checkpoint.get('best_map', 0.0)
                trainer.global_step = checkpoint.get('global_step', 0)
                
                # Try to load scaler state (if using AMP)
                if trainer.use_amp and 'scaler_state_dict' in checkpoint:
                    try:
                        trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                        print("✓ Loaded GradScaler state")
                    except Exception as e:
                        print(f"⚠️  Could not load GradScaler state: {e}")
            
            print(f"\n✓ Resuming from epoch {trainer.start_epoch}")
            print(f"  Best mAP so far: {trainer.best_map:.4f}")
            print(f"  Global step: {trainer.global_step}")
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()