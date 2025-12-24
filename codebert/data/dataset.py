"""PyTorch Dataset for CodeBERT vulnerability detection"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .tokenizer import CodeBERTTokenizer, clean_code


class CodeBERTDataset(Dataset):
    """Dataset for CodeBERT that supports both raw parquet and cached .npz files"""
    
    SPLIT_PATTERNS = {
        'train': 'train-*.parquet',
        'validation': 'validation-*.parquet',
        'test': 'test-*.parquet',
    }
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Optional[CodeBERTTokenizer] = None,
        split: Optional[str] = None,
        func_column: str = 'func_clean',
        label_column: str = 'target',
        max_length: int = 512,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        lazy_load: bool = False,
        chunk_size: int = 5000
    ):
        """Initialize dataset
        
        Args:
            data_path: Path to parquet files or .npz cache file
            tokenizer: CodeBERTTokenizer instance (created if None)
            split: 'train', 'validation', 'test', or None for all
            func_column: Column name for function code ('func' or 'func_clean')
            label_column: Column name for labels
            max_length: Maximum sequence length
            cache_dir: Directory for tokenized cache
            use_cache: Whether to use/create cache
            lazy_load: If True, load data on-demand (memory efficient)
            chunk_size: Chunk size for lazy loading
        """
        self.data_path = Path(data_path)
        self.split = split
        self.func_column = func_column
        self.label_column = label_column
        self.max_length = max_length
        self.use_cache = use_cache
        self.lazy_load = lazy_load
        self.chunk_size = chunk_size
        
        self.tokenizer = tokenizer or CodeBERTTokenizer(
            max_length=max_length,
            cache_dir=cache_dir
        )
        
        self.input_ids: Optional[np.ndarray] = None
        self.attention_mask: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        
        self._df: Optional[pd.DataFrame] = None
        self._length: Optional[int] = None
        self._cache_path: Optional[Path] = None
        
        self._load_data()
    
    def _get_parquet_files(self) -> List[Path]:
        """Get list of parquet files for the split"""
        if self.data_path.suffix == '.npz':
            return []
        
        if self.data_path.is_file():
            return [self.data_path]
        
        if self.split and self.split in self.SPLIT_PATTERNS:
            pattern = self.SPLIT_PATTERNS[self.split]
            files = list(self.data_path.glob(pattern))
        else:
            files = list(self.data_path.glob('*.parquet'))
        
        return sorted(files)
    
    def _get_cache_path(self) -> Path:
        """Generate cache file path"""
        if self.data_path.suffix == '.npz':
            return self.data_path
        
        cache_name = f"codebert_tokenized_{self.split or 'all'}_{self.max_length}.npz"
        return self.data_path / cache_name
    
    def _load_data(self) -> None:
        """Load data from cache or raw files"""
        if self.data_path.suffix == '.npz':
            self._load_from_cache(self.data_path)
            return
        
        self._cache_path = self._get_cache_path()
        
        if self.use_cache and self._cache_path.exists():
            self._load_from_cache(self._cache_path)
            return
        
        self._load_from_parquet()
    
    def _load_from_cache(self, path: Path) -> None:
        """Load tokenized data from .npz cache"""
        data = CodeBERTTokenizer.load_tokenized(path)
        self.input_ids = data['input_ids']
        self.attention_mask = data['attention_mask']
        self.labels = data['labels']
        self._length = len(self.labels)
    
    def _load_from_parquet(self) -> None:
        """Load and tokenize data from parquet files"""
        files = self._get_parquet_files()
        if not files:
            raise FileNotFoundError(f"No parquet files found in {self.data_path}")
        
        if self.lazy_load:
            self._df = self._load_parquet_files(files)
            self._length = len(self._df)
            return
        
        df = self._load_parquet_files(files)
        self._tokenize_dataframe(df)
        
        if self.use_cache and self._cache_path:
            self.tokenizer.save_tokenized(
                self.input_ids,
                self.attention_mask,
                self.labels,
                self._cache_path
            )
    
    def _load_parquet_files(self, files: List[Path]) -> pd.DataFrame:
        """Load and concatenate parquet files"""
        dfs = []
        columns = [self.func_column, self.label_column]
        
        for f in files:
            try:
                df = pd.read_parquet(f, columns=columns)
                dfs.append(df)
            except Exception as e:
                raise IOError(f"Failed to read {f}: {e}") from e
        
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    def _tokenize_dataframe(self, df: pd.DataFrame) -> None:
        """Tokenize entire dataframe"""
        codes = df[self.func_column].fillna('').tolist()
        labels = df[self.label_column].values.astype(np.int64)
        
        tokenized = self.tokenizer.tokenize_batch(codes, clean=True, use_cache=False)
        
        self.input_ids = tokenized['input_ids']
        self.attention_mask = tokenized['attention_mask']
        self.labels = labels
        self._length = len(labels)
    
    def _tokenize_single(self, code: str) -> Dict[str, np.ndarray]:
        """Tokenize a single code sample for lazy loading"""
        return self.tokenizer.tokenize(code, clean=True)
    
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample
        
        Returns:
            Dict with input_ids, attention_mask, labels as tensors
        """
        if self.lazy_load and self._df is not None:
            row = self._df.iloc[idx]
            code = row[self.func_column] or ''
            label = int(row[self.label_column])
            
            tokenized = self._tokenize_single(code)
            
            return {
                'input_ids': torch.tensor(tokenized['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(tokenized['attention_mask'], dtype=torch.long),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
    
    def get_labels(self) -> np.ndarray:
        """Get all labels (for computing class weights)"""
        if self.labels is not None:
            return self.labels
        if self._df is not None:
            return self._df[self.label_column].values.astype(np.int64)
        raise RuntimeError("No data loaded")
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced data"""
        labels = self.get_labels()
        class_counts = np.bincount(labels)
        total = len(labels)
        weights = total / (len(class_counts) * class_counts)
        return torch.tensor(weights, dtype=torch.float)
    
    def get_sample_weights(self) -> np.ndarray:
        """Get per-sample weights for WeightedRandomSampler"""
        labels = self.get_labels()
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        return class_weights[labels]


class CodeBERTCachedDataset(Dataset):
    """Lightweight dataset that loads only from pre-tokenized .npz files"""
    
    def __init__(self, cache_path: Union[str, Path]):
        """Initialize from cached file
        
        Args:
            cache_path: Path to .npz file with input_ids, attention_mask, labels
        """
        self.cache_path = Path(cache_path)
        data = CodeBERTTokenizer.load_tokenized(self.cache_path)
        
        self.input_ids = data['input_ids']
        self.attention_mask = data['attention_mask']
        self.labels = data['labels']
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
    
    def get_labels(self) -> np.ndarray:
        return self.labels
