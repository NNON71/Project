import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb
import yaml
import argparse
import os
from tqdm import tqdm
from pathlib import Path
from typing import Dict
import numpy as np

from models.clip_backbone import OWLViTCLIPBackbone
from mydatasets.clip_dataset import CLIPPretrainingDataset, clip_collate_fn
from utils.metrics import compute_retrieval_metrics

class CLIPBackboneTrainer :
    """Trainer for CLIP Backbone
    Focus: Image-Text Alignment with Contrastive Loss
    """
    
    def __init__(self, config) :
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
        self.model = self._build_model()
        
        # Setup Dataset
        self.train_loader, self.val_loader = self._setup_datasets()
        
        # Setup Optimizer and Scheduler
        self.optimizer, self.scheduler = self._setup_optimizer()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision
        self.use_amp = config['training']['mixed_precision']
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.start_epoch = 0
        self.best_accuracy = 0.0
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
    
    def _build_model(self) -> nn.Module:
        """Build CLIP model"""
        print("\n" + "="*80)
        print("Stage 1: Building CLIP Backbone")
        print("="*80)
        
        model = OWLViTCLIPBackbone(
            d_out=self.config['model']['projection_dim'],
            image_encoder_name=self.config['model']['image_encoder'],
            text_encoder_name=self.config['model']['text_encoder'],
            freeze_vision=self.config['model']['freeze_vision_encoder'],
            freeze_text=self.config['model']['freeze_text_encoder']
        )
        
        model = model.to(self.device)
        
        # Count parameters
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel Statistics:")
        print(f"  Total params: {total:,}")
        print(f"  Trainable params: {trainable:,}")
        print(f"  Frozen params: {total - trainable:,}")
        
        return model
    
    def _setup_datasets(self):
        """Setup train and validation datasets"""
        print("\nSetting up CLIP datasets...")
        
        train_dataset = CLIPPretrainingDataset(
            dataset_name="patomp/thai-mscoco-2014-captions",
            dataset_split="train",
            image_column="image",
            text_column="th_sentences_raw",
            augment=True,
            max_samples=10000
        )
        
        val_dataset = CLIPPretrainingDataset(
            dataset_name="patomp/thai-mscoco-2014-captions",
            dataset_split="validation",
            image_column="image",
            text_column="th_sentences_raw",
            augment=False,
            max_samples=2000
        )
        
        # train_dataset = CLIPPretrainingDataset(
        #     image_dir=self.config['dataset']['train_images'],
        #     captions_file=self.config['dataset']['train_captions'],
        #     image_size=self.config['dataset']['image_size'],
        #     augment=True,
        #     use_thai=True
        # )
        
        # val_dataset = CLIPPretrainingDataset(
        #     image_dir=self.config['dataset']['val_images'],
        #     captions_file=self.config['dataset']['val_captions'],
        #     image_size=self.config['dataset']['image_size'],
        #     augment=False,
        #     use_thai=True
        # )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['dataset']['num_workers'],
            collate_fn=clip_collate_fn,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['dataset']['num_workers'],
            collate_fn=clip_collate_fn,
            pin_memory=True
        )
        
        print(f"✓ Train: {len(train_dataset)} samples, {len(train_loader)} batches")
        print(f"✓ Val: {len(val_dataset)} samples, {len(val_loader)} batches")
        
        return train_loader, val_loader
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        # Separate parameter groups
        param_groups = []
        
        # Vision encoder
        if not self.config['model']['freeze_vision_encoder']:
            param_groups.append({
                'params': self.model.vision_model.parameters(),
                'lr': self.config['training']['learning_rate'] * 0.1,
                'name': 'vision_encoder'
            })
        
        # Text encoder
        if not self.config['model']['freeze_text_encoder']:
            param_groups.append({
                'params': self.model.text_model.parameters(),
                'lr': self.config['training']['learning_rate'] * 0.1,
                'name': 'text_encoder'
            })
        
        # Projection layers
        param_groups.extend([
            {
                'params': self.model.vision_projection.parameters(),
                'lr': self.config['training']['learning_rate'],
                'name': 'vision_projection'
            },
            {
                'params': self.model.text_projection.parameters(),
                'lr': self.config['training']['learning_rate'],
                'name': 'text_projection'
            },
            {
                'params': [self.model.logit_scale],
                'lr': self.config['training']['learning_rate'] * 0.1,
                'name': 'logit_scale'
            }
        ])
        
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config['training']['weight_decay'],
            betas=(0.9, 0.98),
            eps=1e-6
        )
        
        # Cosine scheduler with warmup
        num_training_steps = len(self.train_loader) * self.config['training']['epochs']
        num_warmup_steps = len(self.train_loader) * self.config['training']['warmup_epochs']
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[pg['lr'] for pg in param_groups],
            total_steps=num_training_steps,
            pct_start=num_warmup_steps / num_training_steps,
            anneal_strategy='cos'
        )
        
        print(f"\n✓ Optimizer: AdamW")
        print(f"✓ Scheduler: OneCycleLR (warmup: {num_warmup_steps} steps)")
        
        return optimizer, scheduler
    
    def compute_clip_loss(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CLIP contrastive loss (InfoNCE)
        """
        # Normalize
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # Similarity with temperature
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()
        
        # Labels (diagonal)
        batch_size = image_embeds.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        # Cross-entropy loss
        loss_i2t = self.criterion(logits_per_image, labels)
        loss_t2i = self.criterion(logits_per_text, labels)
        loss = (loss_i2t + loss_t2i) / 2.0
        
        # Accuracy
        with torch.no_grad():
            i2t_acc = (logits_per_image.argmax(1) == labels).float().mean()
            t2i_acc = (logits_per_text.argmax(1) == labels).float().mean()
        
        return {
            'loss': loss,
            'loss_i2t': loss_i2t,
            'loss_t2i': loss_t2i,
            'i2t_acc': i2t_acc,
            't2i_acc': t2i_acc,
            'accuracy': (i2t_acc + t2i_acc) / 2.0,
            'logit_scale': logit_scale
        }
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        
        metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'i2t_acc': 0.0,
            't2i_acc': 0.0
        }
        
        pbar = tqdm(
            self.train_loader,
            desc=f"[Stage 1 - CLIP] Epoch {epoch+1}/{self.config['training']['epochs']}"
        )
        
        for batch_idx, batch in enumerate(pbar):
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    loss_dict = self.compute_clip_loss(
                        outputs['image_embeds'],
                        outputs['text_embeds']
                    )
                
                # Backward
                self.optimizer.zero_grad()
                self.scaler.scale(loss_dict['loss']).backward()
                
                # Gradient clipping
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
                loss_dict = self.compute_clip_loss(
                    outputs['image_embeds'],
                    outputs['text_embeds']
                )
                
                self.optimizer.zero_grad()
                loss_dict['loss'].backward()
                
                if self.config['training']['grad_clip_norm'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip_norm']
                    )
                
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Update metrics
            metrics['loss'] += loss_dict['loss'].item()
            metrics['accuracy'] += loss_dict['accuracy'].item()
            metrics['i2t_acc'] += loss_dict['i2t_acc'].item()
            metrics['t2i_acc'] += loss_dict['t2i_acc'].item()
            
            # Progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['loss'].item():.4f}",
                'acc': f"{loss_dict['accuracy'].item():.3f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}",
                'scale': f"{loss_dict['logit_scale'].item():.2f}"
            })
            
            # Log to W&B
            if self.config['wandb']['enabled'] and \
               batch_idx % self.config['wandb']['log_interval'] == 0:
                wandb.log({
                    'clip_train/loss': loss_dict['loss'].item(),
                    'clip_train/loss_i2t': loss_dict['loss_i2t'].item(),
                    'clip_train/loss_t2i': loss_dict['loss_t2i'].item(),
                    'clip_train/accuracy': loss_dict['accuracy'].item(),
                    'clip_train/i2t_acc': loss_dict['i2t_acc'].item(),
                    'clip_train/t2i_acc': loss_dict['t2i_acc'].item(),
                    'clip_train/logit_scale': loss_dict['logit_scale'].item(),
                    'clip_train/lr': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch,
                    'step': self.global_step
                })
            
            self.global_step += 1
        
        # Average metrics
        for key in metrics:
            metrics[key] /= len(self.train_loader)
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'i2t_acc': 0.0,
            't2i_acc': 0.0
        }
        
        all_image_embeds = []
        all_text_embeds = []
        
        pbar = tqdm(self.val_loader, desc="[CLIP] Validation")
        
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            loss_dict = self.compute_clip_loss(
                outputs['image_embeds'],
                outputs['text_embeds']
            )
            
            metrics['loss'] += loss_dict['loss'].item()
            metrics['accuracy'] += loss_dict['accuracy'].item()
            metrics['i2t_acc'] += loss_dict['i2t_acc'].item()
            metrics['t2i_acc'] += loss_dict['t2i_acc'].item()
            
            # Store embeddings
            all_image_embeds.append(outputs['image_embeds'].cpu())
            all_text_embeds.append(outputs['text_embeds'].cpu())
        
        # Average metrics
        for key in metrics:
            metrics[key] /= len(self.val_loader)
        
        # Compute retrieval metrics
        all_image_embeds = torch.cat(all_image_embeds, dim=0)
        all_text_embeds = torch.cat(all_text_embeds, dim=0)
        
        retrieval_metrics = compute_retrieval_metrics(
            all_image_embeds,
            all_text_embeds,
            k_values=[1, 5, 10]
        )
        metrics.update(retrieval_metrics)
        
        # Log to W&B
        if self.config['wandb']['enabled']:
            wandb.log({
                'clip_val/loss': metrics['loss'],
                'clip_val/accuracy': metrics['accuracy'],
                'clip_val/i2t_acc': metrics['i2t_acc'],
                'clip_val/t2i_acc': metrics['t2i_acc'],
                'clip_val/i2t_R@1': metrics['i2t_R@1'],
                'clip_val/i2t_R@5': metrics['i2t_R@5'],
                'clip_val/i2t_R@10': metrics['i2t_R@10'],
                'clip_val/t2i_R@1': metrics['t2i_R@1'],
                'clip_val/t2i_R@5': metrics['t2i_R@5'],
                'clip_val/t2i_R@10': metrics['t2i_R@10'],
                'epoch': epoch
            })
        
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
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
            print(f"✓ Saved best checkpoint (acc: {self.best_accuracy:.4f})")
        
        # Save epoch
        if (epoch + 1) % self.config['training']['save_interval'] == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{epoch+1}.pth'
            torch.save(checkpoint, epoch_path)
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("Starting CLIP Backbone Training (Stage 1)")
        print("="*80 + "\n")
        
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            
            # Validate
            val_metrics = self.validate(epoch)
            
            print(f"  Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            print(f"    i2t: R@1={val_metrics['i2t_R@1']:.3f}, R@5={val_metrics['i2t_R@5']:.3f}")
            print(f"    t2i: R@1={val_metrics['t2i_R@1']:.3f}, R@5={val_metrics['t2i_R@5']:.3f}")
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_accuracy
            if is_best:
                self.best_accuracy = val_metrics['accuracy']
            
            self.save_checkpoint(epoch, is_best)
        
        print(f"\n{'='*80}")
        print(f"Training Completed! Best Accuracy: {self.best_accuracy:.4f}")
        print(f"{'='*80}\n")
        
        if self.config['wandb']['enabled']:
            wandb.finish()
            
def main():
    parser = argparse.ArgumentParser(description='Train CLIP Backbone (Stage 1)')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = CLIPBackboneTrainer(config)
    
    # Resume if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.start_epoch = checkpoint['epoch'] + 1
        trainer.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        print(f"✓ Resumed from epoch {trainer.start_epoch}")
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()