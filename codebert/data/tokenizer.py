"""CodeBERT Tokenizer wrapper with caching support"""

import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from transformers import AutoTokenizer


def clean_code(code: str) -> str:
    """Remove comments and normalize whitespace from C code
    
    Args:
        code: Raw C source code
        
    Returns:
        Cleaned code with comments removed and whitespace normalized
    """
    code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'[ \t]+', ' ', code)
    code = re.sub(r'\n\s*\n', '\n', code)
    return code.strip()


class CodeBERTTokenizer:
    """Wrapper for CodeBERT tokenizer with caching support"""
    
    MODEL_NAME = "microsoft/codebert-base"
    
    def __init__(
        self, 
        max_length: int = 512,
        cache_dir: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """Initialize tokenizer
        
        Args:
            max_length: Maximum sequence length (default 512)
            cache_dir: Directory for caching tokenized results
            model_name: HuggingFace model name (default: microsoft/codebert-base)
        """
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model_name = model_name or self.MODEL_NAME
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        """Generate cache file path from key"""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.npz"
    
    def _cache_key(self, codes: List[str]) -> str:
        """Generate cache key for a batch of codes"""
        content = f"{self.model_name}|{self.max_length}|" + "|||".join(codes)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def tokenize(
        self, 
        code: str,
        clean: bool = True
    ) -> Dict[str, np.ndarray]:
        """Tokenize a single code snippet
        
        Args:
            code: Source code string
            clean: Whether to clean code before tokenizing
            
        Returns:
            Dict with input_ids and attention_mask as numpy arrays
        """
        if clean:
            code = clean_code(code)
        
        encoded = self.tokenizer(
            code,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }
    
    def tokenize_batch(
        self,
        codes: List[str],
        clean: bool = True,
        use_cache: bool = True
    ) -> Dict[str, np.ndarray]:
        """Tokenize a batch of code snippets
        
        Args:
            codes: List of source code strings
            clean: Whether to clean code before tokenizing
            use_cache: Whether to use disk cache
            
        Returns:
            Dict with input_ids and attention_mask as 2D numpy arrays
        """
        if use_cache and self.cache_dir:
            cache_key = self._cache_key(codes)
            cache_path = self._get_cache_path(cache_key)
            
            if cache_path.exists():
                with np.load(cache_path) as data:
                    return {
                        'input_ids': data['input_ids'],
                        'attention_mask': data['attention_mask']
                    }
        
        if clean:
            codes = [clean_code(c) for c in codes]
        
        encoded = self.tokenizer(
            codes,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        
        result = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
        
        if use_cache and self.cache_dir:
            np.savez_compressed(cache_path, **result)
        
        return result
    
    def save_tokenized(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        labels: np.ndarray,
        path: Union[str, Path]
    ) -> None:
        """Save tokenized data to .npz file
        
        Args:
            input_ids: Token IDs array
            attention_mask: Attention mask array
            labels: Labels array
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    @staticmethod
    def load_tokenized(path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """Load tokenized data from .npz file
        
        Args:
            path: Input file path
            
        Returns:
            Dict with input_ids, attention_mask, labels
        """
        path = Path(path)
        if not path.exists():
            if not path.suffix:
                path = path.with_suffix('.npz')
            if not path.exists():
                raise FileNotFoundError(f"Tokenized file not found: {path}")
        
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.tokenizer.vocab_size
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID"""
        return self.tokenizer.pad_token_id
    
    @property
    def cls_token_id(self) -> int:
        """Get CLS token ID"""
        return self.tokenizer.cls_token_id
    
    @property
    def sep_token_id(self) -> int:
        """Get SEP token ID"""
        return self.tokenizer.sep_token_id
