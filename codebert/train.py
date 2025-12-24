#!/usr/bin/env python3
"""
Training script for CodeBERT vulnerability detection.
"""

import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from config import CodeBERTConfig, get_argument_parser, load_config
from data.dataset import VulnerabilityDataset, create_dataloader
from models.codebert_classifier import CodeBERTClassifier
from utils.metrics import compute_metrics
from utils.trainer import EarlyStopping, Trainer


def setup_logging(output_dir: str) -> logging.Logger:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    log_file = Path(output_dir) / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = get_argument_parser()
    args = parser.parse_args()

    config = load_config(args.config, args)

    logger = setup_logging(config.paths.output_dir)
    logger.info("=" * 60)
    logger.info("CodeBERT Vulnerability Detection Training")
    logger.info("=" * 60)

    set_seed(config.training.seed)
    logger.info(f"Random seed: {config.training.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info(f"Device: {device}, GPUs: {n_gpu}")

    logger.info("Loading configuration:")
    logger.info(json.dumps(config.to_dict(), indent=2))

    Path(config.paths.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    config_save_path = Path(config.paths.output_dir) / "config.yaml"
    import yaml
    with open(config_save_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    logger.info("Initializing model...")
    model = CodeBERTClassifier(
        model_name=config.model.name,
        head_type=config.model.head_type,
        num_labels=config.model.num_labels,
        dropout=config.model.dropout,
        freeze_encoder=config.model.freeze_encoder,
    )

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    logger.info("Loading datasets...")
    train_dataset = VulnerabilityDataset(
        data_path=config.paths.train_data,
        tokenizer=model.tokenizer if not isinstance(model, torch.nn.DataParallel) else model.module.tokenizer,
        max_length=config.data.max_length,
        code_column=config.data.code_column,
        label_column=config.data.label_column,
    )
    valid_dataset = VulnerabilityDataset(
        data_path=config.paths.valid_data,
        tokenizer=model.tokenizer if not isinstance(model, torch.nn.DataParallel) else model.module.tokenizer,
        max_length=config.data.max_length,
        code_column=config.data.code_column,
        label_column=config.data.label_column,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    valid_loader = create_dataloader(
        valid_dataset,
        batch_size=config.training.batch_size * 2,
        shuffle=False,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Valid samples: {len(valid_dataset)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    total_steps = (
        len(train_loader)
        // config.training.gradient_accumulation_steps
        * config.training.max_epochs
    )
    warmup_steps = int(total_steps * config.training.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler() if config.training.fp16 else None

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_grad_norm=1.0,
    )

    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        mode="max",
    )

    logger.info("Starting training...")
    best_metrics = None
    best_epoch = 0

    for epoch in range(1, config.training.max_epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{config.training.max_epochs}")
        logger.info("=" * 60)

        train_loss = trainer.train_epoch(train_loader)
        logger.info(f"Train Loss: {train_loss:.4f}")

        val_loss, val_preds, val_labels = trainer.evaluate(valid_loader)
        val_metrics = compute_metrics(val_labels, val_preds)

        logger.info(f"Valid Loss: {val_loss:.4f}")
        logger.info(f"Valid Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"Valid Precision: {val_metrics['precision']:.4f}")
        logger.info(f"Valid Recall: {val_metrics['recall']:.4f}")
        logger.info(f"Valid F1: {val_metrics['f1']:.4f}")

        if early_stopping(val_metrics["f1"]):
            best_metrics = val_metrics
            best_metrics["epoch"] = epoch
            best_metrics["val_loss"] = val_loss
            best_epoch = epoch

            checkpoint_path = Path(config.paths.checkpoint_dir) / "best_model.pt"
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "metrics": val_metrics,
                    "config": config.to_dict(),
                },
                checkpoint_path,
            )
            logger.info(f"Saved best model to {checkpoint_path}")

        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)

    if best_metrics:
        logger.info(f"\nBest Results (Epoch {best_epoch}):")
        logger.info(f"  Loss:      {best_metrics['val_loss']:.4f}")
        logger.info(f"  Accuracy:  {best_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {best_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {best_metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {best_metrics['f1']:.4f}")

        results_path = Path(config.paths.output_dir) / "results.json"
        with open(results_path, "w") as f:
            json.dump(best_metrics, f, indent=2)
        logger.info(f"\nResults saved to {results_path}")

    if Path(config.paths.test_data).exists():
        logger.info("\nEvaluating on test set...")

        checkpoint = torch.load(
            Path(config.paths.checkpoint_dir) / "best_model.pt",
            map_location=device,
        )
        model_to_load = model.module if isinstance(model, torch.nn.DataParallel) else model
        model_to_load.load_state_dict(checkpoint["model_state_dict"])

        test_dataset = VulnerabilityDataset(
            data_path=config.paths.test_data,
            tokenizer=model_to_load.tokenizer,
            max_length=config.data.max_length,
            code_column=config.data.code_column,
            label_column=config.data.label_column,
        )
        test_loader = create_dataloader(
            test_dataset,
            batch_size=config.training.batch_size * 2,
            shuffle=False,
        )

        test_loss, test_preds, test_labels = trainer.evaluate(test_loader)
        test_metrics = compute_metrics(test_labels, test_preds)

        logger.info(f"\nTest Results:")
        logger.info(f"  Loss:      {test_loss:.4f}")
        logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {test_metrics['f1']:.4f}")

        test_results_path = Path(config.paths.output_dir) / "test_results.json"
        with open(test_results_path, "w") as f:
            json.dump(test_metrics, f, indent=2)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
