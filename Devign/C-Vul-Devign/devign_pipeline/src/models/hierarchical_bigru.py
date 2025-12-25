"""Hierarchical BiGRU model for vulnerability detection with attention extraction.

This is the model architecture used in 03_training_v2.py.
Extended with attention extraction for vulnerability localization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AttentionWeights:
    """Container for attention weights from all layers."""
    global_alpha: torch.Tensor      # [B, T] - global sequence attention
    slice_token_alpha: torch.Tensor # [B, S, L] - per-slice token attention
    slice_level_alpha: torch.Tensor # [B, S] - slice-level attention
    slice_seq_alpha: torch.Tensor   # [B, S] - slice-sequence attention


class HierarchicalBiGRU(nn.Module):
    """Hierarchical BiGRU with global + slice encoders and vulnerability features."""
    
    def __init__(self, vocab_size=238, embed_dim=96, hidden_dim=192, slice_hidden=160,
                 vuln_dim=26, slice_feat_dim=52, gate_init=0.3):
        super().__init__()
        self.slice_hidden = slice_hidden
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(0.3)
        
        self.global_gru = nn.GRU(embed_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.4)
        self.global_attn = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        
        self.slice_gru = nn.GRU(embed_dim, slice_hidden, bidirectional=True, batch_first=True)
        self.slice_attn = nn.Sequential(nn.Linear(slice_hidden*2, slice_hidden), nn.Tanh(), nn.Linear(slice_hidden, 1))
        
        self.slice_seq_gru = nn.GRU(slice_hidden*2, slice_hidden, bidirectional=True, batch_first=True)
        self.slice_seq_attn = nn.Sequential(nn.Linear(slice_hidden*2, slice_hidden), nn.Tanh(), nn.Linear(slice_hidden, 1))
        
        self.slice_feat_mlp = nn.Sequential(nn.Linear(slice_feat_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.4))
        self.slice_fusion = nn.Sequential(
            nn.Linear(slice_hidden*2 + 128, slice_hidden*2),
            nn.GELU(),
            nn.Dropout(0.4)
        )
        self.slice_level_attn = nn.Sequential(nn.Linear(slice_hidden*2, slice_hidden), nn.Tanh(), nn.Linear(slice_hidden, 1))
        
        self.vuln_dim = vuln_dim
        self.vuln_mlp = nn.Sequential(nn.BatchNorm1d(vuln_dim), nn.Linear(vuln_dim, 64), nn.GELU(), nn.Dropout(0.4))
        
        self.feature_gate = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )
        self.gate_strength_raw = nn.Parameter(torch.tensor(gate_init))
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim*2 + slice_hidden*2 + 64), 
            nn.Linear(hidden_dim*2 + slice_hidden*2 + 64, 256), 
            nn.GELU(), nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
    
    @property
    def gate_strength(self):
        return torch.sigmoid(self.gate_strength_raw)
    
    def encode_global(self, ids, mask):
        emb = self.embed_drop(self.embedding(ids))
        out, _ = self.global_gru(emb)
        scores = self.global_attn(out).masked_fill(mask.unsqueeze(-1)==0, -65000.0)
        return (out * F.softmax(scores, dim=1)).sum(dim=1)
    
    def encode_global_with_attn(self, ids: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode global sequence and return attention weights."""
        emb = self.embed_drop(self.embedding(ids))
        out, _ = self.global_gru(emb)
        scores = self.global_attn(out)
        scores = scores.masked_fill(mask.unsqueeze(-1)==0, -65000.0)
        alpha = F.softmax(scores, dim=1)
        g = (out * alpha).sum(dim=1)
        return g, alpha.squeeze(-1)
    
    def encode_slices(self, slice_ids, slice_mask, slice_count, slice_vuln=None, slice_rel=None):
        B, S, L = slice_ids.shape
        emb = self.embed_drop(self.embedding(slice_ids.view(B*S, L)))
        out, _ = self.slice_gru(emb)
        scores = self.slice_attn(out).masked_fill(slice_mask.view(B*S,L).unsqueeze(-1)==0, -65000.0)
        slice_repr = (out * F.softmax(scores, dim=1)).sum(dim=1).view(B, S, -1)
        
        if slice_vuln is not None and slice_rel is not None:
            feat = self.slice_feat_mlp(torch.cat([slice_vuln, slice_rel], dim=-1))
            slice_repr = self.slice_fusion(torch.cat([slice_repr, feat], dim=-1))
        
        s_mask = torch.arange(S, device=slice_count.device).expand(B,S) < slice_count.unsqueeze(1)
        
        s_scores = self.slice_level_attn(slice_repr).masked_fill(~s_mask.unsqueeze(-1), -65000.0)
        slice_attn_repr = (slice_repr * F.softmax(s_scores, dim=1)).sum(dim=1)
        
        slice_repr_masked = slice_repr * s_mask.unsqueeze(-1).float()
        seq_out, _ = self.slice_seq_gru(slice_repr_masked)
        seq_scores = self.slice_seq_attn(seq_out).masked_fill(~s_mask.unsqueeze(-1), -65000.0)
        slice_seq_repr = (seq_out * F.softmax(seq_scores, dim=1)).sum(dim=1)
        
        return 0.5 * (slice_attn_repr + slice_seq_repr)
    
    def encode_slices_with_attn(
        self, 
        slice_ids: torch.Tensor, 
        slice_mask: torch.Tensor, 
        slice_count: torch.Tensor, 
        slice_vuln: Optional[torch.Tensor] = None, 
        slice_rel: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode slices and return all attention weights."""
        B, S, L = slice_ids.shape
        emb = self.embed_drop(self.embedding(slice_ids.view(B*S, L)))
        out, _ = self.slice_gru(emb)
        
        scores_tok = self.slice_attn(out)
        scores_tok = scores_tok.masked_fill(slice_mask.view(B*S,L).unsqueeze(-1)==0, -65000.0)
        alpha_tok = F.softmax(scores_tok, dim=1)
        slice_repr = (out * alpha_tok).sum(dim=1).view(B, S, -1)
        
        alpha_tok = alpha_tok.squeeze(-1).view(B, S, L)
        
        if slice_vuln is not None and slice_rel is not None:
            feat = self.slice_feat_mlp(torch.cat([slice_vuln, slice_rel], dim=-1))
            slice_repr = self.slice_fusion(torch.cat([slice_repr, feat], dim=-1))
        
        s_mask = torch.arange(S, device=slice_count.device).expand(B,S) < slice_count.unsqueeze(1)
        
        s_scores = self.slice_level_attn(slice_repr)
        s_scores = s_scores.masked_fill(~s_mask.unsqueeze(-1), -65000.0)
        alpha_slice_level = F.softmax(s_scores, dim=1).squeeze(-1)
        slice_attn_repr = (slice_repr * alpha_slice_level.unsqueeze(-1)).sum(dim=1)
        
        slice_repr_masked = slice_repr * s_mask.unsqueeze(-1).float()
        seq_out, _ = self.slice_seq_gru(slice_repr_masked)
        seq_scores = self.slice_seq_attn(seq_out)
        seq_scores = seq_scores.masked_fill(~s_mask.unsqueeze(-1), -65000.0)
        alpha_slice_seq = F.softmax(seq_scores, dim=1).squeeze(-1)
        slice_seq_repr = (seq_out * alpha_slice_seq.unsqueeze(-1)).sum(dim=1)
        
        s_repr = 0.5 * (slice_attn_repr + slice_seq_repr)
        
        return s_repr, alpha_tok, alpha_slice_level, alpha_slice_seq
    
    def forward(self, input_ids, attention_mask, slice_input_ids=None, slice_attention_mask=None, 
                slice_count=None, vuln_features=None, slice_vuln_features=None, slice_rel_features=None, **kw):
        g = self.encode_global(input_ids, attention_mask)
        s = self.encode_slices(slice_input_ids, slice_attention_mask, slice_count, slice_vuln_features, slice_rel_features) if slice_input_ids is not None else torch.zeros(g.size(0), self.slice_hidden*2, device=g.device)
        
        if vuln_features is not None:
            v = self.vuln_mlp(vuln_features)
            gate = self.feature_gate(v)
            v = v * (1.0 + self.gate_strength * (gate - 0.5))
        else:
            v = torch.zeros(g.size(0), 64, device=g.device)
        
        h = torch.cat([g, s, v], dim=1)
        logits = self.classifier(h)
        
        return logits
    
    def forward_with_attention(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        slice_input_ids: Optional[torch.Tensor] = None, 
        slice_attention_mask: Optional[torch.Tensor] = None, 
        slice_count: Optional[torch.Tensor] = None, 
        vuln_features: Optional[torch.Tensor] = None, 
        slice_vuln_features: Optional[torch.Tensor] = None, 
        slice_rel_features: Optional[torch.Tensor] = None, 
        **kw
    ) -> Tuple[torch.Tensor, AttentionWeights]:
        """Forward pass that returns logits AND attention weights for localization."""
        g, global_alpha = self.encode_global_with_attn(input_ids, attention_mask)
        
        if slice_input_ids is not None:
            s, slice_token_alpha, slice_level_alpha, slice_seq_alpha = self.encode_slices_with_attn(
                slice_input_ids, slice_attention_mask, slice_count, 
                slice_vuln_features, slice_rel_features
            )
        else:
            B = input_ids.size(0)
            s = torch.zeros(B, self.slice_hidden*2, device=input_ids.device)
            slice_token_alpha = torch.zeros(B, 1, 1, device=input_ids.device)
            slice_level_alpha = torch.zeros(B, 1, device=input_ids.device)
            slice_seq_alpha = torch.zeros(B, 1, device=input_ids.device)
        
        if vuln_features is not None:
            v = self.vuln_mlp(vuln_features)
            gate = self.feature_gate(v)
            v = v * (1.0 + self.gate_strength * (gate - 0.5))
        else:
            v = torch.zeros(g.size(0), 64, device=g.device)
        
        h = torch.cat([g, s, v], dim=1)
        logits = self.classifier(h)
        
        attention_weights = AttentionWeights(
            global_alpha=global_alpha,
            slice_token_alpha=slice_token_alpha,
            slice_level_alpha=slice_level_alpha,
            slice_seq_alpha=slice_seq_alpha,
        )
        
        return logits, attention_weights


def create_hierarchical_model(config: dict) -> HierarchicalBiGRU:
    """Create HierarchicalBiGRU model from config."""
    return HierarchicalBiGRU(
        vocab_size=config.get("vocab_size", 238),
        embed_dim=config.get("embed_dim", 96),
        hidden_dim=config.get("hidden_dim", 192),
        slice_hidden=config.get("slice_hidden", 160),
        vuln_dim=config.get("vuln_dim", 26),
        slice_feat_dim=config.get("slice_feat_dim", 52),
        gate_init=config.get("gate_init", 0.4),
    )
