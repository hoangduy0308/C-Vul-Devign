"""CodeBERT Trainer with dual GPU support for Kaggle T4 environment."""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .metrics import compute_metrics, find_best_threshold, MetricsResult, MetricsTracker
from .utils import set_seed, get_optimizer, get_scheduler, compute_class_weights

logger = logging.getLogger(__name__)


class CodeBERTTrainer:
    """
    Trainer for CodeBERT vulnerability detection model.
    
    Optimized for dual T4 GPU setup on Kaggle:
    - DataParallel for multi-GPU training
    - Mixed precision (fp16) for memory efficiency
    - Gradient accumulation for larger effective batch sizes
    - Learning rate warmup with linear decay
    - Early stopping with checkpoint saving
    
    Default hyperparameters for T4 dual GPU:
    - Batch size: 16 per GPU (32 total)
    - Gradient accumulation: 2 steps (effective batch 64)
    - Learning rate: 2e-5
    - Warmup ratio: 0.1
    - Weight decay: 0.01
    - Max epochs: 10
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_epochs: int = 10,
        gradient_accumulation_steps: int = 2,
        mixed_precision: bool = True,
        early_stopping_patience: int = 3,
        checkpoint_dir: str = "checkpoints",
        pos_weight: Optional[float] = None,
        seed: int = 42,
        use_data_parallel: bool = True,
    ):
        """
        Initialize trainer.
        
        Args:
            model: CodeBERT model for training
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Peak learning rate after warmup
            weight_decay: L2 regularization coefficient
            warmup_ratio: Fraction of training for warmup
            max_epochs: Maximum training epochs
            gradient_accumulation_steps: Steps to accumulate before update
            mixed_precision: Enable fp16 training
            early_stopping_patience: Epochs without improvement to stop
            checkpoint_dir: Directory for saving checkpoints
            pos_weight: Class weight for imbalanced data
            seed: Random seed for reproducibility
            use_data_parallel: Use DataParallel for multi-GPU
        """
        self.seed = seed
        set_seed(seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        
        self.model = model.to(self.device)
        if use_data_parallel and self.n_gpu > 1:
            logger.info(f"Using DataParallel with {self.n_gpu} GPUs")
            self.model = nn.DataParallel(self.model)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.early_stopping_patience = early_stopping_patience
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = get_optimizer(
            self.model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        
        num_training_steps = len(train_loader) * max_epochs // gradient_accumulation_steps
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        
        self.scheduler = get_scheduler(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        self.pos_weight = pos_weight
        if pos_weight is not None:
            weight = torch.tensor([1.0, pos_weight]).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.scaler = GradScaler() if mixed_precision else None
        self.metrics_tracker = MetricsTracker()
        
        self._log_config(learning_rate, weight_decay, warmup_ratio)
    
    def _log_config(
        self,
        learning_rate: float,
        weight_decay: float,
        warmup_ratio: float,
    ) -> None:
        """Log training configuration."""
        effective_batch = (
            self.train_loader.batch_size *
            self.gradient_accumulation_steps *
            max(1, self.n_gpu)
        )
        logger.info(f"Device: {self.device}, GPUs: {self.n_gpu}")
        logger.info(f"Effective batch size: {effective_batch}")
        logger.info(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
        logger.info(f"Warmup ratio: {warmup_ratio}")
        logger.info(f"Mixed precision: {self.mixed_precision}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            leave=False,
        )
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device).long()
            
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss, logits = outputs
                    if hasattr(loss, 'mean'):
                        loss = loss.mean()
                    loss = loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss, logits = outputs
                if hasattr(loss, 'mean'):
                    loss = loss.mean()
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            progress_bar.set_postfix({"loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}"})
        
        return total_loss / num_batches
    
    def evaluate(self, data_loader: Optional[DataLoader] = None) -> Tuple[float, MetricsResult]:
        """
        Evaluate model on validation set.
        
        Args:
            data_loader: DataLoader to evaluate on (defaults to val_loader)
        
        Returns:
            Tuple of (validation_loss, MetricsResult)
        """
        if data_loader is None:
            data_loader = self.val_loader
        
        self.model.eval()
        total_loss = 0.0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device).long()
                
                if self.mixed_precision:
                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        loss, logits = outputs
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss, logits = outputs
                
                if hasattr(loss, 'mean'):
                    loss = loss.mean()
                
                total_loss += loss.item()
                probs = torch.softmax(logits, dim=-1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        best_threshold, _ = find_best_threshold(all_labels, all_probs)
        metrics = compute_metrics(all_labels, all_probs, threshold=best_threshold)
        
        return avg_loss, metrics
    
    def _get_logits(self, outputs) -> torch.Tensor:
        """Extract logits from model outputs."""
        if hasattr(outputs, "logits"):
            return outputs.logits
        elif isinstance(outputs, tuple):
            return outputs[0]
        return outputs
    
    def train(self) -> MetricsTracker:
        """
        Run full training loop.
        
        Returns:
            MetricsTracker with training history
        """
        logger.info(f"Starting training for {self.max_epochs} epochs")
        
        for epoch in range(1, self.max_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_metrics = self.evaluate()
            
            is_best = self.metrics_tracker.update(
                epoch, train_loss, val_loss, val_metrics
            )
            
            log_msg = self.metrics_tracker.log_epoch(epoch, self.max_epochs)
            logger.info(log_msg)
            
            if is_best:
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"New best model! F1: {val_metrics.f1:.4f}")
            
            if self.metrics_tracker.should_stop(self.early_stopping_patience):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info(f"Training complete. Best F1: {self.metrics_tracker.best_f1:.4f} at epoch {self.metrics_tracker.best_epoch}")
        return self.metrics_tracker
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> str:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        
        Returns:
            Path to saved checkpoint
        """
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": self.metrics_tracker.get_summary(),
            "seed": self.seed,
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        filename = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        
        return str(save_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model_to_load = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model_to_load.load_state_dict(checkpoint["model_state_dict"])
        
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]
