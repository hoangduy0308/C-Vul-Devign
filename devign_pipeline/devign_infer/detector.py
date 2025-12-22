"""
Vulnerability Detector - Main inference class for C code vulnerability detection.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from .config import InferenceConfig, ModelConfig


@dataclass
class VulnerabilityResult:
    """Result of vulnerability analysis."""
    vulnerable: bool
    probability: float
    risk_level: str
    confidence: float
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vulnerable": self.vulnerable,
            "probability": self.probability,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "details": self.details
        }


class MultiHeadSelfAttentionPooling(nn.Module):
    """Multi-head self-attention pooling layer."""
    
    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert input_dim % num_heads == 0
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(input_dim, input_dim)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, L, D = x.shape
        
        q = self.query(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        context = self.out_proj(context)
        
        pooled = context.mean(dim=1)
        return pooled


class ImprovedHybridBiGRUVulnDetector(nn.Module):
    """BiGRU model for vulnerability detection."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.use_token_augmentation = getattr(config, 'use_token_augmentation', False)
        self.token_dropout_prob = getattr(config, 'token_dropout_prob', 0.1)
        self.token_mask_prob = getattr(config, 'token_mask_prob', 0.05)
        self.mask_token_id = getattr(config, 'mask_token_id', 1)
        
        self.embedding = nn.Embedding(
            config.vocab_size, 
            config.embed_dim, 
            padding_idx=0
        )
        embed_drop_rate = getattr(config, 'embedding_dropout', 0.15)
        self.embed_dropout = nn.Dropout(embed_drop_rate)
        
        self.gru = nn.GRU(
            config.embed_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            batch_first=True,
            dropout=config.rnn_dropout if config.num_layers > 1 else 0.0
        )
        
        self.rnn_out_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        
        self.use_multihead_attention = getattr(config, 'use_multihead_attention', False)
        if self.use_multihead_attention:
            self.attention = MultiHeadSelfAttentionPooling(
                input_dim=self.rnn_out_dim,
                num_heads=getattr(config, 'num_attention_heads', 4),
                dropout=getattr(config, 'attention_dropout', 0.1),
            )
        else:
            self.attention = nn.Sequential(
                nn.Linear(self.rnn_out_dim, self.rnn_out_dim // 2),
                nn.Tanh(),
                nn.Linear(self.rnn_out_dim // 2, 1, bias=False)
            )
        self.context_dropout = nn.Dropout(0.2)
        
        if config.use_vuln_features:
            self.vuln_bn_in = nn.BatchNorm1d(config.vuln_feature_dim)
            vuln_hidden = getattr(config, 'vuln_feature_hidden_dim', 64)
            self.vuln_mlp = nn.Sequential(
                nn.Linear(config.vuln_feature_dim, vuln_hidden),
                nn.BatchNorm1d(vuln_hidden),
                nn.GELU(),
                nn.Dropout(config.vuln_feature_dropout)
            )
            self.combined_dim = self.rnn_out_dim + vuln_hidden
        else:
            self.combined_dim = self.rnn_out_dim
        
        self.use_layer_norm = getattr(config, 'use_layer_norm', False)
        if self.use_layer_norm:
            self.pre_classifier_ln = nn.LayerNorm(self.combined_dim)
            
        self.classifier = nn.Sequential(
            nn.Linear(self.combined_dim, self.combined_dim // 2),
            nn.BatchNorm1d(self.combined_dim // 2),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(self.combined_dim // 2, 2)
        )
        
    def forward(self, input_ids, attention_mask, vuln_features=None, lengths=None):
        B, L = input_ids.shape
        
        embedded = self.embedding(input_ids)
        embedded = self.embed_dropout(embedded)
        
        use_packing = getattr(self.config, 'use_packed_sequences', False)
        
        if use_packing and lengths is not None:
            lengths_cpu = lengths.cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.gru(packed)
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=L
            )
        else:
            rnn_out, _ = self.gru(embedded)
        
        if self.use_multihead_attention:
            context_vector = self.attention(rnn_out, attention_mask)
        else:
            att_scores = self.attention(rnn_out)
            mask = attention_mask.unsqueeze(-1)
            att_scores = att_scores.masked_fill(mask == 0, -1e4)
            att_weights = F.softmax(att_scores, dim=1)
            context_vector = torch.sum(rnn_out * att_weights, dim=1)
            context_vector = self.context_dropout(context_vector)
        
        if self.config.use_vuln_features and vuln_features is not None:
            feat_out = self.vuln_bn_in(vuln_features)
            feat_out = self.vuln_mlp(feat_out)
            combined = torch.cat([context_vector, feat_out], dim=1)
        else:
            combined = context_vector
        
        if self.use_layer_norm:
            combined = self.pre_classifier_ln(combined)
            
        logits = self.classifier(combined)
        return logits


class VulnerabilityDetector:
    """Main class for vulnerability detection in C code."""
    
    DANGEROUS_APIS = {
        'malloc', 'calloc', 'realloc', 'free', 'alloca',
        'strcpy', 'strcat', 'gets', 'sprintf', 'vsprintf',
        'strncpy', 'strncat', 'strtok', 'strdup', 'strndup',
        'memcpy', 'memmove', 'memset', 'memcmp', 'memchr',
        'printf', 'fprintf', 'sprintf', 'snprintf',
        'scanf', 'fscanf', 'sscanf', 'gets', 'fgets',
        'read', 'fread', 'recv', 'recvfrom',
        'write', 'fwrite', 'send', 'sendto',
        'fopen', 'open', 'system', 'popen', 'exec',
        'getenv', 'atoi', 'atol', 'atof',
    }
    
    C_KEYWORDS = {
        'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
        'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
        'inline', 'int', 'long', 'register', 'restrict', 'return', 'short',
        'signed', 'sizeof', 'static', 'struct', 'switch', 'typedef', 'union',
        'unsigned', 'void', 'volatile', 'while', 'NULL', 'true', 'false'
    }
    
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        config_path: Optional[str] = None,
        device: str = "auto",
        threshold: float = 0.5
    ):
        self.threshold = threshold
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.vocab = self._load_vocab(vocab_path)
        self.model_config = self._load_model_config(config_path)
        self.model = self._load_model(model_path)
        self.model.eval()
        
        self.inference_config = InferenceConfig(
            model_path=model_path,
            vocab_path=vocab_path,
            config_path=config_path,
            device=str(self.device),
            threshold=threshold
        )
    
    def _load_vocab(self, vocab_path: str) -> Dict[str, int]:
        """Load vocabulary from JSON file."""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            if 'token2id' in data:
                return data['token2id']
            return data
        
        return {token: idx for idx, token in enumerate(data)}
    
    def _load_model_config(self, config_path: Optional[str]) -> ModelConfig:
        """Load model configuration."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ModelConfig.from_dict(data)
        
        return ModelConfig(vocab_size=len(self.vocab))
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load PyTorch model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            if 'config' in checkpoint:
                config_data = checkpoint['config']
                if isinstance(config_data, dict):
                    self.model_config = ModelConfig.from_dict(config_data)
        else:
            state_dict = checkpoint
        
        self.model_config.vocab_size = len(self.vocab)
        model = ImprovedHybridBiGRUVulnDetector(self.model_config)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        model.to(self.device)
        return model
    
    def tokenize(self, code: str) -> List[str]:
        """Tokenize C code with hybrid strategy."""
        import re
        
        patterns = [
            (r'"(?:[^"\\]|\\.)*"', 'STR'),
            (r"'(?:[^'\\]|\\.)*'", 'CHAR'),
            (r'/\*[\s\S]*?\*/', None),
            (r'//[^\n]*', None),
            (r'0[xX][0-9a-fA-F]+[uUlL]*', 'NUM'),
            (r'\d+\.\d*[fFlL]?|\.\d+[fFlL]?', 'FLOAT'),
            (r'\d+[uUlL]*', 'NUM'),
            (r'[a-zA-Z_][a-zA-Z0-9_]*', 'ID'),
            (r'->|<<|>>|<=|>=|==|!=|&&|\|\|', 'OP'),
            (r'[+\-*/%&|^~<>=!]', 'OP'),
            (r'[(){}\[\];,.:?#]', 'PUNCT'),
            (r'\s+', None),
        ]
        
        tokens = []
        pos = 0
        
        while pos < len(code):
            matched = False
            
            for pattern, token_type in patterns:
                match = re.match(pattern, code[pos:])
                if match:
                    text = match.group()
                    pos += len(text)
                    
                    if token_type is None:
                        matched = True
                        break
                    
                    if token_type == 'ID':
                        if text in self.DANGEROUS_APIS:
                            tokens.append(text)
                        elif text in self.C_KEYWORDS:
                            tokens.append(text)
                        else:
                            tokens.append('ID')
                    elif token_type in ('STR', 'CHAR', 'NUM', 'FLOAT'):
                        tokens.append(token_type)
                    else:
                        tokens.append(text)
                    
                    matched = True
                    break
            
            if not matched:
                pos += 1
        
        return tokens
    
    def vectorize(self, tokens: List[str], max_length: int = 512) -> tuple:
        """Convert tokens to model input."""
        bos_id = self.vocab.get('<BOS>', 2)
        eos_id = self.vocab.get('<EOS>', 3)
        unk_id = self.vocab.get('<UNK>', 1)
        pad_id = self.vocab.get('<PAD>', 0)
        
        ids = [bos_id]
        for token in tokens:
            ids.append(self.vocab.get(token, unk_id))
        ids.append(eos_id)
        
        if len(ids) > max_length:
            ids = ids[:max_length-1] + [eos_id]
        
        attention_mask = [1] * len(ids)
        
        while len(ids) < max_length:
            ids.append(pad_id)
            attention_mask.append(0)
        
        return ids, attention_mask
    
    def extract_vuln_features(self, code: str) -> List[float]:
        """Extract vulnerability-related features from code."""
        features = []
        
        for api in sorted(self.DANGEROUS_APIS)[:26]:
            count = code.count(api)
            features.append(min(count, 10) / 10.0)
        
        while len(features) < 26:
            features.append(0.0)
        
        return features[:26]
    
    def analyze(
        self,
        code: str,
        return_details: bool = True
    ) -> VulnerabilityResult:
        """
        Analyze C code for vulnerabilities.
        
        Args:
            code: C source code string
            return_details: Include detailed analysis info
            
        Returns:
            VulnerabilityResult with prediction
        """
        tokens = self.tokenize(code)
        ids, attention_mask = self.vectorize(tokens)
        vuln_features = self.extract_vuln_features(code)
        
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask_t = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
        vuln_features_t = torch.tensor([vuln_features], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask_t, vuln_features_t)
            probs = F.softmax(logits, dim=1)
            vuln_prob = probs[0, 1].item()
        
        vulnerable = vuln_prob >= self.threshold
        risk_level = self.inference_config.get_risk_level(vuln_prob)
        confidence = abs(vuln_prob - 0.5) * 2
        
        details = {}
        if return_details:
            detected_apis = [api for api in self.DANGEROUS_APIS if api in code]
            details = {
                "num_tokens": len(tokens),
                "dangerous_apis_found": detected_apis,
                "threshold_used": self.threshold,
                "model_device": str(self.device)
            }
        
        return VulnerabilityResult(
            vulnerable=vulnerable,
            probability=vuln_prob,
            risk_level=risk_level,
            confidence=confidence,
            details=details
        )
    
    def analyze_file(self, file_path: str) -> VulnerabilityResult:
        """Analyze a C source file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
        
        result = self.analyze(code)
        result.details["file_path"] = file_path
        return result
    
    def analyze_batch(
        self,
        codes: List[str],
        batch_size: int = 32
    ) -> List[VulnerabilityResult]:
        """Analyze multiple code snippets in batch."""
        results = []
        
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i:i+batch_size]
            batch_tokens = [self.tokenize(c) for c in batch_codes]
            batch_vectorized = [self.vectorize(t) for t in batch_tokens]
            batch_vuln_feats = [self.extract_vuln_features(c) for c in batch_codes]
            
            input_ids = torch.tensor(
                [v[0] for v in batch_vectorized],
                dtype=torch.long, device=self.device
            )
            attention_masks = torch.tensor(
                [v[1] for v in batch_vectorized],
                dtype=torch.long, device=self.device
            )
            vuln_features = torch.tensor(
                batch_vuln_feats,
                dtype=torch.float32, device=self.device
            )
            
            with torch.no_grad():
                logits = self.model(input_ids, attention_masks, vuln_features)
                probs = F.softmax(logits, dim=1)
            
            for j, (code, tokens) in enumerate(zip(batch_codes, batch_tokens)):
                vuln_prob = probs[j, 1].item()
                vulnerable = vuln_prob >= self.threshold
                risk_level = self.inference_config.get_risk_level(vuln_prob)
                confidence = abs(vuln_prob - 0.5) * 2
                
                results.append(VulnerabilityResult(
                    vulnerable=vulnerable,
                    probability=vuln_prob,
                    risk_level=risk_level,
                    confidence=confidence,
                    details={"num_tokens": len(tokens)}
                ))
        
        return results
