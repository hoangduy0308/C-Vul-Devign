"""
Vulnerability Detector - Inference with dynamic vocab tokenization.

Model Architecture: HierarchicalSliceVulnDetector (v2)
Vocab: Dynamic vocab size (auto-detected from vocab file)
"""

import os
import sys
import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from .config import InferenceConfig


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


# ============================================================================
# Model Architecture (matching checkpoint)
# ============================================================================

class HierarchicalSliceVulnDetector(nn.Module):
    """Hierarchical Slice-based Vulnerability Detector (v2)."""
    
    def __init__(
        self,
        vocab_size: int = 238,
        embed_dim: int = 96,
        global_hidden: int = 192,
        global_layers: int = 2,
        slice_hidden: int = 160,
        slice_layers: int = 1,
        vuln_feature_dim: int = 26,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.gate_strength_raw = nn.Parameter(torch.tensor(0.0))
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.global_gru = nn.GRU(
            embed_dim, global_hidden, 
            num_layers=global_layers,
            bidirectional=True, 
            batch_first=True,
            dropout=dropout if global_layers > 1 else 0
        )
        self.global_attn = nn.Sequential(
            nn.Linear(global_hidden * 2, global_hidden),
            nn.Tanh(),
            nn.Linear(global_hidden, 1, bias=True)
        )
        
        self.slice_gru = nn.GRU(
            embed_dim, slice_hidden,
            num_layers=slice_layers,
            bidirectional=True,
            batch_first=True
        )
        self.slice_attn = nn.Sequential(
            nn.Linear(slice_hidden * 2, slice_hidden),
            nn.Tanh(),
            nn.Linear(slice_hidden, 1, bias=True)
        )
        
        self.slice_seq_gru = nn.GRU(
            slice_hidden * 2, slice_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.slice_seq_attn = nn.Sequential(
            nn.Linear(slice_hidden * 2, slice_hidden),
            nn.Tanh(),
            nn.Linear(slice_hidden, 1, bias=True)
        )
        
        slice_feat_input = vuln_feature_dim * 2
        self.slice_feat_mlp = nn.Sequential(
            nn.Linear(slice_feat_input, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        slice_fusion_input = slice_hidden * 2 + 128
        self.slice_fusion = nn.Sequential(
            nn.Linear(slice_fusion_input, slice_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.slice_level_attn = nn.Sequential(
            nn.Linear(slice_hidden * 2, slice_hidden),
            nn.Tanh(),
            nn.Linear(slice_hidden, 1, bias=True)
        )
        
        self.vuln_mlp = nn.Sequential(
            nn.BatchNorm1d(vuln_feature_dim),
            nn.Linear(vuln_feature_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.feature_gate = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )
        
        classifier_input = global_hidden * 2 + slice_hidden * 2 + 64
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(classifier_input),
            nn.Linear(classifier_input, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )
    
    def _attention_pool(self, rnn_out, attn_layer, mask=None):
        scores = attn_layer(rnn_out)
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)
        pooled = (rnn_out * weights).sum(dim=1)
        return pooled
    
    def forward(
        self,
        global_ids: torch.Tensor,
        global_mask: torch.Tensor,
        slice_ids: torch.Tensor = None,
        slice_mask: torch.Tensor = None,
        slice_features: torch.Tensor = None,
        vuln_features: torch.Tensor = None,
    ):
        B = global_ids.shape[0]
        
        global_emb = self.embedding(global_ids)
        global_out, _ = self.global_gru(global_emb)
        global_vec = self._attention_pool(global_out, self.global_attn, global_mask)
        
        if slice_ids is not None and slice_ids.numel() > 0:
            S = slice_ids.shape[1]
            L_slice = slice_ids.shape[2]
            
            slice_ids_flat = slice_ids.view(B * S, L_slice)
            slice_mask_flat = slice_mask.view(B * S, L_slice) if slice_mask is not None else None
            
            slice_emb = self.embedding(slice_ids_flat)
            slice_out, _ = self.slice_gru(slice_emb)
            slice_vec = self._attention_pool(slice_out, self.slice_attn, slice_mask_flat)
            slice_vec = slice_vec.view(B, S, -1)
            
            slice_seq_out, _ = self.slice_seq_gru(slice_vec)
            slice_seq_vec = self._attention_pool(slice_seq_out, self.slice_seq_attn)
            
            if slice_features is not None:
                slice_feat_flat = slice_features.view(B * S, -1)
                slice_feat_out = self.slice_feat_mlp(slice_feat_flat)
                slice_feat_out = slice_feat_out.view(B, S, -1)
                
                fused = torch.cat([slice_vec, slice_feat_out], dim=-1)
                fused = self.slice_fusion(fused.view(B * S, -1)).view(B, S, -1)
                slice_final = self._attention_pool(fused, self.slice_level_attn)
            else:
                slice_final = slice_seq_vec
        else:
            slice_final = torch.zeros(B, 320, device=global_ids.device)
        
        if vuln_features is not None:
            vuln_out = self.vuln_mlp(vuln_features)
            gate = self.feature_gate(vuln_out)
            vuln_out = vuln_out * gate
        else:
            vuln_out = torch.zeros(B, 64, device=global_ids.device)
        
        combined = torch.cat([global_vec, slice_final, vuln_out], dim=-1)
        logits = self.classifier(combined)
        
        return logits


# ============================================================================
# Dynamic Vocab Tokenization (supports any vocab size)
# ============================================================================

C_KEYWORDS: Set[str] = {
    'if', 'else', 'for', 'while', 'return', 'break', 'case', 'continue',
    'switch', 'goto', 'sizeof', 'const', 'static', 'void', 'int', 'char',
    'unsigned', 'struct', 'long', 'float', 'true', 'false', 'double',
    'signed', 'short', 'enum', 'union', 'typedef', 'extern', 'NULL',
    'default', 'do', 'inline', 'volatile', 'register', 'restrict',
    '_Bool', '_Complex', '_Imaginary', 'nullptr', 'auto'
}

DANGEROUS_APIS: Set[str] = {
    'fprintf', 'memcpy', 'memset', 'strcmp', 'printf', 'strlen', 'snprintf',
    'close', 'free', 'strncmp', 'memcmp', 'sscanf', 'write', 'strcpy',
    'memmove', 'read', 'fclose', 'fopen', 'open', 'malloc', 'sprintf',
    'strncpy', 'fwrite', 'puts', 'fread', 'strcat', 'vsnprintf', 'fgets',
    'strncat', 'realloc', 'calloc', 'alloca', 'fscanf', 'memchr', 'fputs',
    'getchar', 'strchr', 'strdup', 'strndup', 'strstr', 'strrchr',
    'getenv', 'system', 'popen', 'pclose', 'gets', 'scanf', 'getline',
    'recv', 'send', 'recvfrom', 'sendto', 'recvmsg', 'sendmsg'
}

PRESERVED_TOKENS: Set[str] = C_KEYWORDS | DANGEROUS_APIS


def _get_special_token_id(vocab: Dict[str, int], names: List[str], default: int) -> int:
    """Get special token ID, trying multiple possible names."""
    for name in names:
        if name in vocab:
            return vocab[name]
    return default


class DynamicVocabTokenizer:
    """
    Dynamic tokenizer that works with any vocab size.
    
    Supports vocab formats:
    - With angle brackets: <PAD>, <UNK>, <BOS>, <EOS>, <SEP>
    - Without angle brackets: PAD, UNK, BOS, EOS, SEP
    
    Normalization rules:
    - Numbers -> NUM/num (auto-detected from vocab)
    - Hex numbers -> NUM_HEX or NUM
    - Floats -> FLOAT or NUM
    - Strings -> STR  
    - Chars -> CHAR
    - C keywords -> preserved if in vocab
    - Dangerous APIs -> preserved if in vocab
    - Other identifiers -> lookup in vocab, else UNK
    """
    
    _PATTERNS = [
        (r'//[^\n]*', None),
        (r'/\*[\s\S]*?\*/', None),
        (r'"(?:[^"\\]|\\.)*"', 'STR'),
        (r"'(?:[^'\\]|\\.)*'", 'CHAR'),
        (r'0[xX][0-9a-fA-F]+[uUlL]*', 'NUM_HEX'),
        (r'\d+\.\d*(?:[eE][+-]?\d+)?[fFlL]*', 'FLOAT'),
        (r'\.\d+(?:[eE][+-]?\d+)?[fFlL]*', 'FLOAT'),
        (r'\d+[uUlL]*', 'NUM'),
        (r'[a-zA-Z_][a-zA-Z0-9_]*', 'ID'),
        (r'->|<<|>>|<=|>=|==|!=|&&|\|\|', 'OP'),
        (r'\+\+|--|[+\-*/%&|^~<>=!?:]', 'OP'),
        (r'[(){}[\];,.#\\]', 'PUNCT'),
        (r'\s+', None),
    ]
    
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self._compiled = [(re.compile(p, re.DOTALL), t) for p, t in self._PATTERNS]
        
        self.pad_id = _get_special_token_id(vocab, ['<PAD>', 'PAD'], 0)
        self.unk_id = _get_special_token_id(vocab, ['<UNK>', 'UNK'], 1)
        self.bos_id = _get_special_token_id(vocab, ['<BOS>', 'BOS'], 2)
        self.eos_id = _get_special_token_id(vocab, ['<EOS>', 'EOS'], 3)
        self.sep_id = _get_special_token_id(vocab, ['<SEP>', 'SEP'], 4)
        
        self.num_token = 'NUM' if 'NUM' in vocab else 'num'
        self.str_token = 'STR' if 'STR' in vocab else 'string'
        self.char_token = 'CHAR' if 'CHAR' in vocab else 'char'
        self.float_token = 'FLOAT' if 'FLOAT' in vocab else self.num_token
        self.hex_token = 'NUM_HEX' if 'NUM_HEX' in vocab else self.num_token
        self.unk_token = '<UNK>' if '<UNK>' in vocab else 'UNK'
    
    def tokenize(self, code: str) -> List[str]:
        """Tokenize and normalize code."""
        raw_tokens = self._raw_tokenize(code)
        return self._normalize(raw_tokens)
    
    def _raw_tokenize(self, code: str) -> List[Tuple[str, str]]:
        tokens = []
        pos = 0
        while pos < len(code):
            matched = False
            for pattern, ttype in self._compiled:
                m = pattern.match(code, pos)
                if m:
                    text = m.group()
                    pos = m.end()
                    if ttype is not None:
                        tokens.append((text, ttype))
                    matched = True
                    break
            if not matched:
                pos += 1
        return tokens
    
    def _normalize(self, raw_tokens: List[Tuple[str, str]]) -> List[str]:
        """Normalize tokens using vocab lookup."""
        normalized = []
        
        for text, ttype in raw_tokens:
            if ttype == 'STR':
                normalized.append(self.str_token)
            elif ttype == 'CHAR':
                normalized.append(self.char_token)
            elif ttype == 'NUM':
                if text in self.vocab:
                    normalized.append(text)
                else:
                    normalized.append(self.num_token)
            elif ttype == 'NUM_HEX':
                if text.lower() in self.vocab:
                    normalized.append(text.lower())
                elif text in self.vocab:
                    normalized.append(text)
                else:
                    normalized.append(self.hex_token)
            elif ttype == 'FLOAT':
                normalized.append(self.float_token)
            elif ttype in ('OP', 'PUNCT'):
                if text in self.vocab:
                    normalized.append(text)
                else:
                    normalized.append(self.unk_token)
            elif ttype == 'ID':
                if text in self.vocab:
                    normalized.append(text)
                elif text in PRESERVED_TOKENS and text in self.vocab:
                    normalized.append(text)
                else:
                    normalized.append(self.unk_token)
            else:
                if text in self.vocab:
                    normalized.append(text)
                else:
                    normalized.append(self.unk_token)
        
        return normalized
    
    def vectorize(self, tokens: List[str], max_len: int = 512) -> Tuple[List[int], List[int]]:
        """Convert tokens to IDs with BOS/EOS and padding."""
        ids = [self.bos_id]
        for t in tokens:
            ids.append(self.vocab.get(t, self.unk_id))
        ids.append(self.eos_id)
        
        if len(ids) > max_len:
            ids = ids[:max_len - 1] + [self.eos_id]
        
        mask = [1] * len(ids)
        pad_len = max_len - len(ids)
        ids += [self.pad_id] * pad_len
        mask += [0] * pad_len
        
        return ids, mask


# Backward compatibility alias
Vocab238Tokenizer = DynamicVocabTokenizer


class InferencePreprocessor:
    """Preprocessing pipeline for inference."""
    
    def __init__(
        self,
        vocab: Dict[str, int],
        max_seq_length: int = 512,
        max_slices: int = 6,
        slice_max_len: int = 256,
        slice_window: int = 15,
    ):
        self.tokenizer = DynamicVocabTokenizer(vocab)
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.max_slices = max_slices
        self.slice_max_len = slice_max_len
        self.slice_window = slice_window
    
    def extract_slices(self, code: str) -> List[str]:
        """Extract slices around dangerous API calls."""
        lines = code.split('\n')
        dangerous_lines = []
        
        for i, line in enumerate(lines):
            for api in DANGEROUS_APIS:
                if re.search(rf'\b{api}\s*\(', line):
                    dangerous_lines.append(i)
                    break
        
        if not dangerous_lines:
            return [code]
        
        slices = []
        for line_idx in dangerous_lines[:self.max_slices]:
            start = max(0, line_idx - self.slice_window)
            end = min(len(lines), line_idx + self.slice_window + 1)
            slice_code = '\n'.join(lines[start:end])
            slices.append(slice_code)
        
        return slices
    
    def extract_vuln_features(self, code: str) -> List[float]:
        """Extract 26 vulnerability features."""
        features = []
        lines = code.split('\n')
        
        features.append(min(len(lines), 500) / 500.0)
        features.append(min(code.count(';'), 100) / 100.0)
        dangerous_count = sum(1 for api in DANGEROUS_APIS if api in code)
        features.append(min(dangerous_count, 20) / 20.0)
        
        while len(features) < 26:
            features.append(0.0)
        
        return features[:26]
    
    def preprocess(self, code: str) -> Dict[str, Any]:
        """Full preprocessing for inference."""
        global_tokens = self.tokenizer.tokenize(code)
        global_ids, global_mask = self.tokenizer.vectorize(global_tokens, self.max_seq_length)
        
        slices = self.extract_slices(code)
        slice_ids_list = []
        slice_mask_list = []
        slice_features_list = []
        
        for s in slices[:self.max_slices]:
            s_tokens = self.tokenizer.tokenize(s)
            s_ids, s_mask = self.tokenizer.vectorize(s_tokens, self.slice_max_len)
            slice_ids_list.append(s_ids)
            slice_mask_list.append(s_mask)
            
            s_feat = self.extract_vuln_features(s)
            s_rel = [0.0] * 26
            slice_features_list.append(s_feat + s_rel)
        
        pad_id = self.tokenizer.pad_id
        while len(slice_ids_list) < self.max_slices:
            slice_ids_list.append([pad_id] * self.slice_max_len)
            slice_mask_list.append([0] * self.slice_max_len)
            slice_features_list.append([0.0] * 52)
        
        vuln_features = self.extract_vuln_features(code)
        
        return {
            'global_ids': global_ids,
            'global_mask': global_mask,
            'slice_ids': slice_ids_list,
            'slice_mask': slice_mask_list,
            'slice_features': slice_features_list,
            'vuln_features': vuln_features,
            'num_slices': len(slices),
            'num_tokens': len(global_tokens),
        }


class VulnerabilityDetector:
    """Main class for vulnerability detection."""
    
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        config_path: Optional[str] = None,
        device: str = "auto",
        threshold: float = 0.5,
    ):
        self.threshold = threshold
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.vocab = self._load_vocab(vocab_path)
        self.config = self._load_config(config_path)
        
        self.model = HierarchicalSliceVulnDetector(
            vocab_size=len(self.vocab),
            embed_dim=self.config.get('embed_dim', 96),
            global_hidden=self.config.get('global_hidden', 192),
            global_layers=self.config.get('global_layers', 2),
            slice_hidden=self.config.get('slice_hidden', 160),
            slice_layers=self.config.get('slice_layers', 1),
            vuln_feature_dim=self.config.get('vuln_feature_dim', 26),
            dropout=self.config.get('dropout', 0.3)
        )
        
        self._load_weights(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.preprocessor = InferencePreprocessor(
            vocab=self.vocab,
            max_seq_length=self.config.get('max_len', 512),
            max_slices=self.config.get('max_slices', 6),
            slice_max_len=self.config.get('slice_max_len', 256),
        )
        
        self.inference_config = InferenceConfig(
            model_path=model_path,
            vocab_path=vocab_path,
            config_path=config_path,
            device=str(self.device),
            threshold=threshold
        )
    
    def _load_vocab(self, vocab_path: str) -> Dict[str, int]:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            if 'token2id' in data:
                return data['token2id']
            return data
        return {t: i for i, t in enumerate(data)}
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_weights(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                state_dict[k[7:]] = v
            else:
                state_dict[k] = v
        
        self.model.load_state_dict(state_dict, strict=False)
    
    def analyze(self, code: str, return_details: bool = True) -> VulnerabilityResult:
        """Analyze code for vulnerabilities."""
        data = self.preprocessor.preprocess(code)
        
        global_ids = torch.tensor([data['global_ids']], dtype=torch.long, device=self.device)
        global_mask = torch.tensor([data['global_mask']], dtype=torch.long, device=self.device)
        slice_ids = torch.tensor([data['slice_ids']], dtype=torch.long, device=self.device)
        slice_mask = torch.tensor([data['slice_mask']], dtype=torch.long, device=self.device)
        slice_features = torch.tensor([data['slice_features']], dtype=torch.float32, device=self.device)
        vuln_features = torch.tensor([data['vuln_features']], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            logits = self.model(
                global_ids, global_mask,
                slice_ids, slice_mask,
                slice_features, vuln_features
            )
            probs = F.softmax(logits, dim=1)
            vuln_prob = probs[0, 1].item()
        
        vulnerable = vuln_prob >= self.threshold
        risk_level = self.inference_config.get_risk_level(vuln_prob)
        confidence = abs(vuln_prob - 0.5) * 2
        
        details = {}
        if return_details:
            details = {
                "num_slices": data['num_slices'],
                "num_tokens": data['num_tokens'],
                "dangerous_apis_found": [api for api in DANGEROUS_APIS if api in code],
                "threshold_used": self.threshold,
                "model_device": str(self.device),
                "vocab_size": len(self.vocab),
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
    
    def analyze_batch(self, codes: List[str], batch_size: int = 16) -> List[VulnerabilityResult]:
        """Analyze multiple code snippets with true batch processing."""
        results = []
        
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i:i + batch_size]
            batch_data = [self.preprocessor.preprocess(code) for code in batch_codes]
            
            global_ids = torch.tensor([d['global_ids'] for d in batch_data], dtype=torch.long, device=self.device)
            global_mask = torch.tensor([d['global_mask'] for d in batch_data], dtype=torch.long, device=self.device)
            slice_ids = torch.tensor([d['slice_ids'] for d in batch_data], dtype=torch.long, device=self.device)
            slice_mask = torch.tensor([d['slice_mask'] for d in batch_data], dtype=torch.long, device=self.device)
            slice_features = torch.tensor([d['slice_features'] for d in batch_data], dtype=torch.float32, device=self.device)
            vuln_features = torch.tensor([d['vuln_features'] for d in batch_data], dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                logits = self.model(
                    global_ids, global_mask,
                    slice_ids, slice_mask,
                    slice_features, vuln_features
                )
                probs = F.softmax(logits, dim=1)
                vuln_probs = probs[:, 1].cpu().tolist()
            
            for vuln_prob in vuln_probs:
                vulnerable = vuln_prob >= self.threshold
                risk_level = self.inference_config.get_risk_level(vuln_prob)
                confidence = abs(vuln_prob - 0.5) * 2
                results.append(VulnerabilityResult(
                    vulnerable=vulnerable,
                    probability=vuln_prob,
                    risk_level=risk_level,
                    confidence=confidence,
                    details={}
                ))
        
        return results
