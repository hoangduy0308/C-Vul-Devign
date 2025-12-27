"""
Vectorization Strategy Module

Provides a common interface for vectorizing tokenized code across different tokenizer types.
Reduces if/elif complexity in preprocessing scripts by using the Strategy pattern.
"""

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm


@dataclass
class VectorizationConfig:
    """Configuration for vectorization."""
    max_len: int = 512
    truncation_strategy: str = 'head_tail'
    head_tokens: int = 192
    tail_tokens: int = 319
    batch_size: int = 500


@dataclass
class VectorizationResult:
    """Result of batch vectorization."""
    input_ids: np.ndarray
    attention_mask: np.ndarray
    unk_positions: List[List[int]]
    stats: Dict


class VectorizationStrategy(ABC):
    """Abstract base class for vectorization strategies."""
    
    def __init__(self, config: VectorizationConfig = None):
        self.config = config or VectorizationConfig()
    
    @abstractmethod
    def vectorize_single(
        self,
        tokens: List[str],
        vocab: Dict[str, int]
    ) -> Tuple[List[int], List[int], List[int], List[str]]:
        """
        Vectorize a single token sequence.
        
        Args:
            tokens: List of token strings
            vocab: Token to ID mapping
            
        Returns:
            Tuple of (input_ids, attention_mask, unk_positions, unk_tokens)
            where unk_tokens contains the actual token strings that were mapped to UNK.
        """
        pass
    
    def vectorize_batch(
        self,
        tokens_list: List[List[str]],
        vocab: Dict[str, int],
        show_progress: bool = True
    ) -> VectorizationResult:
        """
        Vectorize a batch of token sequences with statistics.
        
        Args:
            tokens_list: List of tokenized samples
            vocab: Token to ID mapping
            show_progress: Whether to show progress bar
            
        Returns:
            VectorizationResult with input_ids, attention_mask, unk_positions, and stats
        """
        if len(tokens_list) == 0:
            return VectorizationResult(
                input_ids=np.zeros((0, self.config.max_len), dtype=np.int32),
                attention_mask=np.zeros((0, self.config.max_len), dtype=np.int32),
                unk_positions=[],
                stats={
                    'total_tokens': 0,
                    'total_unks': 0,
                    'unk_rate': 0.0,
                    'top_unk_tokens': [],
                    'truncated_samples': 0,
                }
            )
        
        all_input_ids = []
        all_attention_masks = []
        all_unk_positions = []
        
        total_tokens = 0
        total_unks = 0
        truncated_count = 0
        unk_tokens = Counter()
        
        iterator = tokens_list
        if show_progress:
            iterator = tqdm(tokens_list, desc="Vectorizing")
        
        for tokens in iterator:
            if len(tokens) > self.config.max_len:
                truncated_count += 1
            
            result = self.vectorize_single(tokens, vocab)
            # Handle both old 3-tuple and new 4-tuple return formats
            if len(result) == 4:
                input_ids, attention_mask, unk_positions, unk_token_strs = result
            else:
                input_ids, attention_mask, unk_positions = result
                unk_token_strs = []
            
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_unk_positions.append(unk_positions)
            
            actual_len = sum(attention_mask)
            total_tokens += actual_len
            total_unks += len(unk_positions)
            
            # Use the actual UNK token strings directly
            for tok in unk_token_strs:
                unk_tokens[tok] += 1
        
        stats = {
            'total_tokens': total_tokens,
            'total_unks': total_unks,
            'unk_rate': total_unks / total_tokens if total_tokens > 0 else 0,
            'top_unk_tokens': unk_tokens.most_common(50),
            'truncated_samples': truncated_count,
            'truncated_ratio': truncated_count / len(tokens_list) if tokens_list else 0,
        }
        
        return VectorizationResult(
            input_ids=np.array(all_input_ids, dtype=np.int32),
            attention_mask=np.array(all_attention_masks, dtype=np.int32),
            unk_positions=all_unk_positions,
            stats=stats
        )


class HeadTailVectorizationStrategy(VectorizationStrategy):
    """
    Vectorization with head-tail truncation and UNK tracking.
    Used by 'preserve' and 'optimized' tokenizers.
    """
    
    def _apply_truncation(self, tokens: List[str]) -> List[str]:
        """Apply truncation strategy to tokens."""
        max_len = self.config.max_len
        
        if len(tokens) <= max_len:
            return tokens
        
        if self.config.truncation_strategy == 'front':
            return tokens[-max_len:]
        
        if self.config.truncation_strategy == 'head_tail':
            head_tokens = self.config.head_tokens
            tail_tokens = self.config.tail_tokens
            
            # Guard against edge case where max_len is too small
            limit = max(0, max_len - 1)  # Reserve 1 for SEP
            effective_head = min(head_tokens, limit)
            effective_tail = min(tail_tokens, limit - effective_head)
            
            if len(tokens) <= effective_head + effective_tail:
                return tokens
            
            head = tokens[:effective_head]
            tail = tokens[-effective_tail:] if effective_tail > 0 else []
            return head + ['SEP'] + tail
        
        # 'back' - default
        return tokens[:max_len]
    
    def vectorize_single(
        self,
        tokens: List[str],
        vocab: Dict[str, int]
    ) -> Tuple[List[int], List[int], List[int], List[str]]:
        """Vectorize with UNK position tracking.
        
        Returns:
            Tuple of (input_ids, attention_mask, unk_positions, unk_tokens)
            where unk_tokens contains the actual token strings that were mapped to UNK.
        """
        unk_id = vocab.get('UNK', 1)
        pad_id = vocab.get('PAD', 0)
        max_len = self.config.max_len
        
        truncated = self._apply_truncation(tokens)
        
        input_ids = []
        unk_positions = []
        unk_tokens = []  # Track actual UNK token strings
        
        for i, tok in enumerate(truncated):
            if tok in vocab:
                input_ids.append(vocab[tok])
            else:
                input_ids.append(unk_id)
                unk_positions.append(i)
                unk_tokens.append(tok)  # Store the actual token
        
        actual_len = len(input_ids)
        if actual_len < max_len:
            input_ids.extend([pad_id] * (max_len - actual_len))
        
        attention_mask = [1] * actual_len + [0] * (max_len - actual_len)
        
        return input_ids, attention_mask, unk_positions, unk_tokens


class SimpleVectorizationStrategy(VectorizationStrategy):
    """
    Simple tail-truncation vectorization without UNK tracking.
    Used by 'canonical' tokenizer.
    """
    
    def vectorize_single(
        self,
        tokens: List[str],
        vocab: Dict[str, int]
    ) -> Tuple[List[int], List[int], List[int], List[str]]:
        """Vectorize with simple tail truncation."""
        unk_id = vocab.get('UNK', 1)
        pad_id = vocab.get('PAD', 0)
        max_len = self.config.max_len
        
        truncated = tokens[:max_len]
        input_ids = [vocab.get(tok, unk_id) for tok in truncated]
        actual_len = len(input_ids)
        
        if actual_len < max_len:
            input_ids.extend([pad_id] * (max_len - actual_len))
        
        attention_mask = [1] * actual_len + [0] * (max_len - actual_len)
        
        # Track UNK positions and tokens
        unk_positions = []
        unk_tokens = []
        for i, tok in enumerate(truncated):
            if tok not in vocab:
                unk_positions.append(i)
                unk_tokens.append(tok)
        
        return input_ids, attention_mask, unk_positions, unk_tokens


def get_vectorization_strategy(
    tokenizer_type: str,
    config: VectorizationConfig = None
) -> VectorizationStrategy:
    """
    Factory function to get the appropriate vectorization strategy.
    
    Args:
        tokenizer_type: One of 'preserve', 'optimized', 'subtoken', or 'canonical'
        config: Vectorization configuration
        
    Returns:
        VectorizationStrategy instance
    """
    if tokenizer_type in ('preserve', 'optimized', 'subtoken'):
        return HeadTailVectorizationStrategy(config)
    elif tokenizer_type == 'canonical':
        return SimpleVectorizationStrategy(config)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
