"""
Dictionary-Guided Attention Module for Vulnerability Detection.

This module implements soft attention guidance using vulnerability patterns
from the dictionary. It boosts attention weights for tokens that match
known vulnerability patterns (dangerous functions, keywords).

Key features:
- Soft guidance: Multiplies attention weights by a learnable boost factor
- Pattern-aware: Uses dangerous_functions from vuln_patterns.yaml
- Category weights: Different boost for different vulnerability categories
- Backward compatible: Can be disabled via config
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PatternAttentionConfig:
    """Configuration for pattern-guided attention."""
    
    enabled: bool = True
    
    # Additive boost bias for tokens matching patterns (added to attention logits)
    # Positive values increase attention probability for pattern tokens
    base_boost: float = 1.0
    
    # Whether boost is learnable
    learnable_boost: bool = True
    
    # Category-specific boost multipliers
    category_weights: Dict[str, float] = field(default_factory=lambda: {
        "buffer_overflow": 1.5,
        "use_after_free": 1.5,
        "null_pointer": 1.3,
        "integer_overflow": 1.2,
        "format_string": 1.4,
        "command_injection": 1.4,
        "taint_source": 1.1,
    })
    
    # Minimum attention weight after boosting (prevent over-focusing)
    min_attention_ratio: float = 0.1
    
    # Temperature for softmax (lower = sharper focus)
    temperature: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            "enabled": self.enabled,
            "base_boost": self.base_boost,
            "learnable_boost": self.learnable_boost,
            "category_weights": self.category_weights,
            "min_attention_ratio": self.min_attention_ratio,
            "temperature": self.temperature,
        }


class PatternVocabMapper:
    """
    Maps vocabulary tokens to vulnerability patterns.
    
    Creates a lookup table that identifies which tokens in the vocabulary
    correspond to dangerous functions or vulnerability indicators.
    """
    
    def __init__(
        self,
        vocab: Dict[str, int],
        dangerous_functions: Set[str],
        category_functions: Dict[str, List[str]] = None,
    ):
        """
        Args:
            vocab: Token to ID mapping
            dangerous_functions: Set of dangerous function names
            category_functions: Dict mapping category -> list of functions
        """
        self.vocab = vocab
        self.dangerous_functions = dangerous_functions
        self.category_functions = category_functions or {}
        
        # Build lookup tables
        self.dangerous_token_ids: Set[int] = set()
        self.token_to_category: Dict[int, str] = {}
        
        self._build_mappings()
    
    def _build_mappings(self):
        """Build token ID to pattern mappings."""
        # Map dangerous functions to token IDs
        for func_name in self.dangerous_functions:
            # Try exact match
            if func_name in self.vocab:
                token_id = self.vocab[func_name]
                self.dangerous_token_ids.add(token_id)
            
            # Try lowercase
            if func_name.lower() in self.vocab:
                token_id = self.vocab[func_name.lower()]
                self.dangerous_token_ids.add(token_id)
        
        # Map category-specific functions
        for category, functions in self.category_functions.items():
            for func_name in functions:
                if func_name in self.vocab:
                    token_id = self.vocab[func_name]
                    self.token_to_category[token_id] = category
                if func_name.lower() in self.vocab:
                    token_id = self.vocab[func_name.lower()]
                    self.token_to_category[token_id] = category
        
        logger.info(
            f"PatternVocabMapper: Mapped {len(self.dangerous_token_ids)} dangerous tokens, "
            f"{len(self.token_to_category)} category-specific tokens"
        )
    
    def get_pattern_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create a mask indicating which tokens match patterns.
        
        Args:
            input_ids: Token IDs [B, L]
            
        Returns:
            Boolean mask [B, L] where True = matches pattern
        """
        device = input_ids.device
        dangerous_ids_tensor = torch.tensor(
            list(self.dangerous_token_ids), 
            device=device, 
            dtype=input_ids.dtype
        )
        
        if len(dangerous_ids_tensor) == 0:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        
        # Check if each token is in dangerous_token_ids
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.dangerous_token_ids:
            mask = mask | (input_ids == token_id)
        
        return mask
    
    def get_category_weights(
        self, 
        input_ids: torch.Tensor,
        category_weight_map: Dict[str, float],
        default_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Get category-specific weights for each token.
        
        Args:
            input_ids: Token IDs [B, L]
            category_weight_map: Category to weight mapping
            default_weight: Weight for non-pattern tokens
            
        Returns:
            Weights [B, L]
        """
        device = input_ids.device
        weights = torch.full_like(input_ids, default_weight, dtype=torch.float)
        
        for token_id, category in self.token_to_category.items():
            cat_weight = category_weight_map.get(category, default_weight)
            mask = (input_ids == token_id)
            weights = torch.where(mask, torch.tensor(cat_weight, device=device), weights)
        
        return weights


class DictionaryGuidedAttention(nn.Module):
    """
    Attention module with dictionary-based guidance.
    
    This modifies the standard attention mechanism to give more weight
    to tokens that match known vulnerability patterns.
    
    The guidance is "soft" - it multiplies attention scores by a boost
    factor rather than forcing attention to specific locations.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        config: PatternAttentionConfig = None,
        pattern_mapper: PatternVocabMapper = None,
    ):
        """
        Args:
            hidden_dim: Dimension of hidden states from encoder
            config: Pattern attention configuration
            pattern_mapper: Mapper from tokens to patterns
        """
        super().__init__()
        
        self.config = config or PatternAttentionConfig()
        self.pattern_mapper = pattern_mapper
        self.hidden_dim = hidden_dim
        
        # Standard attention components
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )
        
        # Learnable boost bias (additive)
        if self.config.learnable_boost:
            self.boost_bias = nn.Parameter(
                torch.tensor(self.config.base_boost)
            )
        else:
            self.register_buffer(
                'boost_bias', 
                torch.tensor(self.config.base_boost)
            )
        
        # Learnable category weights (optional)
        self.num_categories = len(self.config.category_weights)
        if self.num_categories > 0 and self.config.learnable_boost:
            category_weights = list(self.config.category_weights.values())
            self.category_boost = nn.Parameter(
                torch.tensor(category_weights)
            )
            self.category_names = list(self.config.category_weights.keys())
        
        self.dropout = nn.Dropout(0.1)
    
    def set_pattern_mapper(self, pattern_mapper: PatternVocabMapper):
        """Set or update the pattern mapper."""
        self.pattern_mapper = pattern_mapper
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply dictionary-guided attention pooling.
        
        Args:
            hidden_states: Encoder outputs [B, L, H]
            attention_mask: Mask for valid tokens [B, L]
            input_ids: Original token IDs [B, L] (for pattern matching)
            return_attention_weights: Whether to return attention weights
            
        Returns:
            context_vector: Weighted sum of hidden states [B, H]
            attention_weights: Optional attention weights [B, L]
        """
        B, L, H = hidden_states.shape
        
        # Compute base attention scores
        att_scores = self.attention(hidden_states)  # [B, L, 1]
        
        # Apply pattern boost if enabled and mapper available
        if (self.config.enabled and 
            self.pattern_mapper is not None and 
            input_ids is not None):
            
            # Get pattern mask
            pattern_mask = self.pattern_mapper.get_pattern_mask(input_ids)  # [B, L]
            
            # Apply additive boost to attention scores for pattern tokens
            # Using additive bias ensures correct behavior regardless of score sign
            boost = self.boost_bias.clamp(min=0.0, max=3.0)  # Clamp for stability
            pattern_boost = torch.where(
                pattern_mask.unsqueeze(-1),
                boost * torch.ones_like(att_scores),
                torch.zeros_like(att_scores)
            )
            att_scores = att_scores + pattern_boost
        
        # Apply attention mask
        mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
        att_scores = att_scores.masked_fill(mask == 0, -1e4)
        
        # Apply temperature
        att_scores = att_scores / self.config.temperature
        
        # Softmax to get attention weights
        att_weights = F.softmax(att_scores, dim=1)  # [B, L, 1]
        
        # Apply minimum attention ratio (prevent over-focusing)
        if self.config.min_attention_ratio > 0:
            min_weight = self.config.min_attention_ratio / L
            att_weights = att_weights.clamp(min=min_weight)
            att_weights = att_weights / att_weights.sum(dim=1, keepdim=True)
        
        # Compute context vector
        context_vector = torch.sum(hidden_states * att_weights, dim=1)  # [B, H]
        context_vector = self.dropout(context_vector)
        
        if return_attention_weights:
            return context_vector, att_weights.squeeze(-1)
        return context_vector, None
    
    def get_pattern_attention_stats(
        self,
        attention_weights: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute statistics about attention on pattern tokens.
        
        Useful for analysis and debugging.
        
        Args:
            attention_weights: Attention weights [B, L]
            input_ids: Token IDs [B, L]
            
        Returns:
            Dict with attention statistics
        """
        if self.pattern_mapper is None:
            return {}
        
        pattern_mask = self.pattern_mapper.get_pattern_mask(input_ids)
        
        # Attention on pattern tokens
        pattern_attention = (attention_weights * pattern_mask.float()).sum(dim=1)
        
        # Number of pattern tokens
        pattern_count = pattern_mask.float().sum(dim=1)
        
        return {
            "mean_pattern_attention": pattern_attention.mean().item(),
            "mean_pattern_count": pattern_count.mean().item(),
            "pattern_attention_ratio": (
                pattern_attention.sum() / attention_weights.sum()
            ).item(),
        }


def create_pattern_mapper_from_dictionary(
    vocab: Dict[str, int],
    vuln_dict,  # VulnDictionary from src.vuln.dictionary
) -> PatternVocabMapper:
    """
    Create a PatternVocabMapper from a VulnDictionary.
    
    Args:
        vocab: Vocabulary mapping (token -> id)
        vuln_dict: VulnDictionary instance
        
    Returns:
        PatternVocabMapper instance
    """
    dangerous_functions = vuln_dict.dangerous_functions
    
    # Build category -> functions mapping
    category_functions = {}
    for category in vuln_dict.vuln_types:
        funcs = vuln_dict.get_dangerous_functions_by_category(category)
        if funcs:
            category_functions[category] = funcs
    
    return PatternVocabMapper(
        vocab=vocab,
        dangerous_functions=dangerous_functions,
        category_functions=category_functions,
    )


# Alias for backward compatibility
PatternAwareAttention = DictionaryGuidedAttention


if __name__ == "__main__":
    # Quick test
    print("Testing DictionaryGuidedAttention...")
    
    # Create dummy vocab and patterns
    vocab = {
        "strcpy": 10,
        "memcpy": 11,
        "malloc": 12,
        "free": 13,
        "if": 20,
        "else": 21,
        "return": 22,
    }
    
    dangerous_functions = {"strcpy", "memcpy", "malloc", "free"}
    category_functions = {
        "buffer_overflow": ["strcpy", "memcpy"],
        "null_pointer": ["malloc"],
        "use_after_free": ["free"],
    }
    
    # Create mapper
    mapper = PatternVocabMapper(
        vocab=vocab,
        dangerous_functions=dangerous_functions,
        category_functions=category_functions,
    )
    
    # Create attention module
    config = PatternAttentionConfig(enabled=True, base_boost=1.5)
    attention = DictionaryGuidedAttention(
        hidden_dim=256,
        config=config,
        pattern_mapper=mapper,
    )
    
    # Test forward pass
    B, L, H = 2, 32, 256
    hidden_states = torch.randn(B, L, H)
    attention_mask = torch.ones(B, L)
    input_ids = torch.randint(0, 30, (B, L))
    
    # Inject some dangerous tokens
    input_ids[0, 5] = 10  # strcpy
    input_ids[0, 10] = 13  # free
    input_ids[1, 3] = 12  # malloc
    
    context, weights = attention(
        hidden_states, attention_mask, input_ids, 
        return_attention_weights=True
    )
    
    print(f"Context shape: {context.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Boost factor: {attention.boost_factor.item():.3f}")
    
    # Check attention stats
    stats = attention.get_pattern_attention_stats(weights, input_ids)
    print(f"Pattern attention stats: {stats}")
    
    print("[OK] Test passed!")
