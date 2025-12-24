"""DataLoader utilities for CodeBERT training"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import CodeBERTDataset, CodeBERTCachedDataset
from .tokenizer import CodeBERTTokenizer


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader
    
    Args:
        batch: List of sample dicts from dataset
        
    Returns:
        Batched dict with stacked tensors
    """
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch])
    }


def create_dataloader(
    dataset: Union[CodeBERTDataset, CodeBERTCachedDataset],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    use_weighted_sampler: bool = False,
    pin_memory: bool = True
) -> DataLoader:
    """Create a DataLoader for a dataset
    
    Args:
        dataset: CodeBERT dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle (ignored if using weighted sampler)
        num_workers: Number of worker processes
        use_weighted_sampler: Use WeightedRandomSampler for imbalanced data
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        Configured DataLoader
    """
    sampler = None
    
    if use_weighted_sampler:
        sample_weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False
    )


def create_dataloaders(
    data_path: Union[str, Path],
    tokenizer: Optional[CodeBERTTokenizer] = None,
    batch_size: int = 32,
    max_length: int = 512,
    func_column: str = 'func_clean',
    label_column: str = 'target',
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    use_weighted_sampler: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    splits: Optional[List[str]] = None
) -> Dict[str, DataLoader]:
    """Create DataLoaders for train/validation/test splits
    
    Args:
        data_path: Path to parquet files directory
        tokenizer: Shared tokenizer instance
        batch_size: Batch size for all loaders
        max_length: Maximum sequence length
        func_column: Column name for function code
        label_column: Column name for labels
        cache_dir: Directory for tokenized cache
        use_cache: Whether to use/create cache
        use_weighted_sampler: Use weighted sampling for training
        num_workers: Number of data loading workers
        pin_memory: Pin memory for GPU transfer
        splits: List of splits to load (default: all available)
        
    Returns:
        Dict mapping split names to DataLoaders
    """
    data_path = Path(data_path)
    splits = splits or ['train', 'validation', 'test']
    
    tokenizer = tokenizer or CodeBERTTokenizer(
        max_length=max_length,
        cache_dir=cache_dir
    )
    
    dataloaders = {}
    
    for split in splits:
        cache_file = data_path / f"codebert_tokenized_{split}_{max_length}.npz"
        
        if use_cache and cache_file.exists():
            dataset = CodeBERTCachedDataset(cache_file)
        else:
            dataset = CodeBERTDataset(
                data_path=data_path,
                tokenizer=tokenizer,
                split=split,
                func_column=func_column,
                label_column=label_column,
                max_length=max_length,
                cache_dir=cache_dir,
                use_cache=use_cache
            )
        
        is_train = split == 'train'
        
        dataloaders[split] = create_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=is_train and not use_weighted_sampler,
            num_workers=num_workers,
            use_weighted_sampler=is_train and use_weighted_sampler,
            pin_memory=pin_memory
        )
    
    return dataloaders


def create_train_val_test_loaders(
    data_path: Union[str, Path],
    batch_size: int = 32,
    max_length: int = 512,
    use_weighted_sampler: bool = True,
    num_workers: int = 0,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Convenience function to get train, val, test loaders as tuple
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    loaders = create_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        max_length=max_length,
        use_weighted_sampler=use_weighted_sampler,
        num_workers=num_workers,
        splits=['train', 'validation', 'test'],
        **kwargs
    )
    
    return loaders['train'], loaders['validation'], loaders['test']


def get_class_weights(data_path: Union[str, Path], split: str = 'train') -> torch.Tensor:
    """Get class weights for loss function
    
    Args:
        data_path: Path to parquet files
        split: Which split to compute weights from
        
    Returns:
        Tensor of class weights
    """
    dataset = CodeBERTDataset(
        data_path=data_path,
        split=split,
        use_cache=True,
        lazy_load=True
    )
    return dataset.get_class_weights()
