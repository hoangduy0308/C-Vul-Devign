"""Model inference module for HierarchicalBiGRU vulnerability detection.

This module matches the training pipeline from 03_training_v2.py.
Extended with attention-based vulnerability localization.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import numpy as np
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from devign_pipeline.src.models.hierarchical_bigru import HierarchicalBiGRU, AttentionWeights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATHS = [
    Path("models/best_v2_seed42.pt"),
    Path("models/best_v2_seed1042.pt"),
    Path("models/best_v2_seed2042.pt"),
]
VOCAB_PATH = Path("models/vocab.json")
CONFIG_PATH = Path("models/config.json")
FEATURE_STATS_PATH = Path("models/feature_stats.json")
ENSEMBLE_CONFIG_PATH = Path("ensemble_config.json")

MAX_LEN = 512
NUM_SLICES = 6
SLICE_LEN = 256
VULN_DIM = 26


class PredictionRequest(BaseModel):
    code: str


class PredictionResponse(BaseModel):
    vulnerable: bool
    score: float
    threshold: float
    confidence: str


@dataclass
class VulnerableLocation:
    """A highlighted vulnerable location in the code."""
    line: int
    score: float
    normalized_score: float
    code_snippet: str = ""
    tokens: List[str] = field(default_factory=list)


@dataclass 
class LocalizationResult:
    """Full localization result with prediction and highlights."""
    prediction: PredictionResponse
    highlights: List[VulnerableLocation]
    

C_KEYWORDS = {
    'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
    'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
    'int', 'long', 'register', 'return', 'short', 'signed', 'sizeof', 'static',
    'struct', 'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while',
    'NULL', 'true', 'false', 'nullptr'
}

COMMON_STDLIB_FUNCS = {
    'printf', 'scanf', 'malloc', 'calloc', 'realloc', 'free',
    'memcpy', 'memset', 'memmove', 'memcmp', 'strlen', 'strcpy',
    'strncpy', 'strcat', 'strncat', 'strcmp', 'strncmp', 'strchr',
    'strrchr', 'strstr', 'sprintf', 'snprintf', 'sscanf', 'fprintf',
    'fscanf', 'fopen', 'fclose', 'fread', 'fwrite', 'fgets', 'fputs',
    'exit', 'abort', 'atoi', 'atol', 'atof', 'strtol', 'getchar', 'putchar',
    'gets', 'puts', 'getenv', 'system', 'assert', 'perror', 'open', 'close',
    'read', 'write', 'alloca', 'vprintf', 'vsnprintf', 'vsprintf'
}

DANGEROUS_FUNCTIONS = [
    'strcpy', 'strcat', 'gets', 'sprintf', 'memcpy', 'memmove',
    'scanf', 'fscanf', 'sscanf', 'vsprintf', 'vprintf'
]


@dataclass
class TokenInfo:
    """Token with position information."""
    text: str
    line: int
    start_pos: int
    end_pos: int


def tokenize_c_code_with_positions(code: str) -> Tuple[List[str], List[int]]:
    """Tokenize C code and return tokens with their line numbers."""
    lines = code.split('\n')
    line_starts = [0]
    for line in lines:
        line_starts.append(line_starts[-1] + len(line) + 1)
    
    def pos_to_line(pos: int) -> int:
        for i, start in enumerate(line_starts):
            if i + 1 < len(line_starts) and pos < line_starts[i + 1]:
                return i + 1
        return len(lines)
    
    clean_code = re.sub(r'/\*.*?\*/', lambda m: ' ' * len(m.group()), code, flags=re.DOTALL)
    clean_code = re.sub(r'//.*?$', lambda m: ' ' * len(m.group()), clean_code, flags=re.MULTILINE)
    
    patterns = [
        (r'"(?:[^"\\]|\\.)*"', 'STR'),
        (r"'(?:[^'\\]|\\.)*'", 'CHAR'),
        (r'0[xX][0-9a-fA-F]+[uUlL]*', 'NUM'),
        (r'0[bB][01]+[uUlL]*', 'NUM'),
        (r'\d+\.?\d*(?:[eE][+-]?\d+)?[fFlLuU]*', 'NUM'),
        (r'\.\d+(?:[eE][+-]?\d+)?[fFlL]*', 'NUM'),
        (r'\.\.\.', None),
        (r'::', None),
        (r'->', None),
        (r'\+\+|--', None),
        (r'<<=|>>=', None),
        (r'<<|>>', None),
        (r'<=|>=|==|!=', None),
        (r'&&|\|\|', None),
        (r'[+\-*/%&|^~!=<>]=', None),
        (r'[+\-*/%&|^~!=<>?:#]', None),
        (r'[a-zA-Z_][a-zA-Z0-9_]*', 'ID'),
        (r'[{}()\[\];,.]', None),
    ]
    
    compiled = [(re.compile(p), t) for p, t in patterns]
    
    tokens = []
    token_lines = []
    pos = 0
    
    while pos < len(clean_code):
        if clean_code[pos].isspace():
            pos += 1
            continue
        
        matched = False
        for pattern, token_type in compiled:
            match = pattern.match(clean_code, pos)
            if match:
                text = match.group()
                line = pos_to_line(pos)
                
                if token_type == 'STR':
                    tokens.append('STR')
                elif token_type == 'CHAR':
                    tokens.append('CHAR')
                elif token_type == 'NUM':
                    tokens.append('NUM')
                else:
                    tokens.append(text)
                
                token_lines.append(line)
                pos = match.end()
                matched = True
                break
        
        if not matched:
            pos += 1
    
    return tokens, token_lines


def normalize_tokens(tokens: List[str]) -> List[str]:
    """Normalize tokens: variables -> VAR_N, non-stdlib functions -> FUNC_N."""
    normalized = []
    var_map: Dict[str, str] = {}
    func_map: Dict[str, str] = {}
    var_counter = 0
    func_counter = 0
    
    for i, token in enumerate(tokens):
        if token in C_KEYWORDS:
            normalized.append(token)
        elif token in ('NUM', 'STR', 'CHAR'):
            normalized.append(token)
        elif token in COMMON_STDLIB_FUNCS:
            normalized.append(token)
        elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
            next_token = tokens[i + 1] if i + 1 < len(tokens) else None
            is_func = next_token == '('
            
            if is_func:
                if token not in func_map:
                    func_map[token] = f'FUNC_{func_counter}'
                    func_counter += 1
                normalized.append(func_map[token])
            else:
                if token not in var_map:
                    var_map[token] = f'VAR_{var_counter}'
                    var_counter += 1
                normalized.append(var_map[token])
        else:
            normalized.append(token)
    
    return normalized


def extract_vuln_features(code: str, tokens: List[str]) -> np.ndarray:
    """Extract 26 vulnerability features matching training."""
    feats = []
    
    loc = code.count('\n') + 1
    feats.append(loc)
    
    stmt_count = code.count(';')
    feats.append(stmt_count)
    
    dangerous_count = 0
    for func in DANGEROUS_FUNCTIONS:
        dangerous_count += len(re.findall(rf'\b{func}\s*\(', code))
    feats.append(dangerous_count)
    
    if_count = len(re.findall(r'\bif\s*\(', code))
    feats.append(max(0, dangerous_count - if_count))
    
    feats.append(dangerous_count / max(1, stmt_count) if stmt_count else 0)
    
    ptr_deref = code.count('->') + code.count('*')
    feats.append(ptr_deref)
    
    null_checks = len(re.findall(r'\bNULL\b', code)) + len(re.findall(r'!=\s*NULL', code))
    feats.append(max(0, ptr_deref - null_checks))
    
    feats.append(max(0, ptr_deref - null_checks) / max(1, ptr_deref) if ptr_deref else 0)
    
    array_access = code.count('[')
    feats.append(array_access)
    
    bounds_checks = len(re.findall(r'<\s*\w+', code)) + len(re.findall(r'>\s*0', code))
    feats.append(max(0, array_access - bounds_checks))
    
    feats.append(max(0, array_access - bounds_checks) / max(1, array_access) if array_access else 0)
    
    malloc_count = len(re.findall(r'\bmalloc\s*\(', code)) + len(re.findall(r'\bcalloc\s*\(', code))
    feats.append(malloc_count)
    
    free_count = len(re.findall(r'\bfree\s*\(', code))
    feats.append(max(0, malloc_count - free_count))
    
    feats.append(max(0, malloc_count - free_count) / max(1, malloc_count) if malloc_count else 0)
    
    feats.append(free_count)
    
    feats.append(max(0, free_count - null_checks))
    
    feats.append(max(0, free_count - null_checks) / max(1, free_count) if free_count else 0)
    
    func_calls = len(re.findall(r'\b\w+\s*\(', code))
    feats.append(max(0, func_calls - if_count))
    
    feats.append(max(0, func_calls - if_count) / max(1, func_calls) if func_calls else 0)
    
    feats.append(null_checks)
    
    feats.append(bounds_checks)
    
    feats.append((null_checks + bounds_checks) / max(1, dangerous_count + ptr_deref + array_access))
    
    feats.append(dangerous_count / max(1, loc))
    
    feats.append(ptr_deref / max(1, loc))
    
    feats.append(array_access / max(1, loc))
    
    feats.append(null_checks / max(1, loc))
    
    if len(feats) < VULN_DIM:
        feats.extend([0.0] * (VULN_DIM - len(feats)))
    else:
        feats = feats[:VULN_DIM]
    
    return np.array(feats, dtype=np.float32)


class ModelWrapper:
    def __init__(
        self,
        model_paths: List[Path] = None,
        vocab_path: Path = VOCAB_PATH,
        config_path: Path = CONFIG_PATH,
        feature_stats_path: Path = FEATURE_STATS_PATH,
        ensemble_config_path: Path = ENSEMBLE_CONFIG_PATH
    ) -> None:
        self.model_paths = model_paths or MODEL_PATHS
        self.vocab_path = vocab_path
        self.config_path = config_path
        self.feature_stats_path = feature_stats_path
        self.ensemble_config_path = ensemble_config_path
        
        self.data_config = self._load_data_config()
        self.ensemble_config = self._load_ensemble_config()
        self.vocab = self._load_vocab()
        self.feature_stats = self._load_feature_stats()
        self.models = self._load_models()
        self.threshold = float(self.ensemble_config.get("optimal_threshold", 0.37))
        
        self.max_len = self.data_config.get("max_len", MAX_LEN)
        self.num_slices = self.data_config.get("max_slices", NUM_SLICES)
        self.slice_len = self.data_config.get("slice_max_len", SLICE_LEN)

    def _load_data_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"vocab_size": 238, "max_len": 512, "max_slices": 6, "slice_max_len": 256}

    def _load_ensemble_config(self) -> Dict[str, Any]:
        if self.ensemble_config_path.exists():
            return json.loads(self.ensemble_config_path.read_text())
        return {"optimal_threshold": 0.37}

    def _load_vocab(self) -> Dict[str, int]:
        if self.vocab_path.exists():
            data = json.loads(self.vocab_path.read_text())
            return data.get("token2id", {})
        return {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
    
    def _load_feature_stats(self) -> Dict[str, Any]:
        if self.feature_stats_path.exists():
            data = json.loads(self.feature_stats_path.read_text())
            return data.get("feature_stats", {})
        return {}

    def _load_single_model(self, model_path: Path) -> HierarchicalBiGRU:
        """Load a single model from path."""
        vocab_size = self.data_config.get("vocab_size", 238)
        
        model = HierarchicalBiGRU(
            vocab_size=vocab_size,
            embed_dim=96,
            hidden_dim=192,
            slice_hidden=160,
            vuln_dim=VULN_DIM,
            slice_feat_dim=52,
            gate_init=float(self.ensemble_config.get("gate_init", 0.4)),
        )
        
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        
        model.to(DEVICE)
        model.eval()
        return model

    def _load_models(self) -> List[HierarchicalBiGRU]:
        """Load ensemble of models."""
        models = []
        for path in self.model_paths:
            if path.exists():
                models.append(self._load_single_model(path))
        
        if not models:
            models.append(self._load_single_model(self.model_paths[0]))
        
        return models

    def _preprocess_with_mapping(self, code: str) -> Tuple[Dict[str, torch.Tensor], List[str], List[int], List[int], List[Tuple[int, int]]]:
        """Preprocess and return token-to-position mappings for localization."""
        tokens, token_lines = tokenize_c_code_with_positions(code)
        normalized = normalize_tokens(tokens)
        
        pad_id = self.vocab.get("<PAD>", 0)
        unk_id = self.vocab.get("<UNK>", 1)
        bos_id = self.vocab.get("<BOS>", 2)
        eos_id = self.vocab.get("<EOS>", 3)
        
        global_ids = [bos_id]
        global_token_indices = []
        for i, t in enumerate(normalized[:self.max_len - 2]):
            global_ids.append(self.vocab.get(t, unk_id))
            global_token_indices.append(i)
        global_ids.append(eos_id)
        
        input_ids = torch.full((1, self.max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((1, self.max_len), dtype=torch.float)
        length = min(len(global_ids), self.max_len)
        input_ids[0, :length] = torch.tensor(global_ids[:length], dtype=torch.long)
        attention_mask[0, :length] = 1.0
        
        slices = []
        slice_token_indices = []
        for start in range(0, len(normalized), self.slice_len - 2):
            if len(slices) >= self.num_slices:
                break
            end = min(start + self.slice_len - 2, len(normalized))
            slices.append(normalized[start:end])
            slice_token_indices.append(list(range(start, end)))
        
        while len(slices) < self.num_slices:
            slices.append([])
            slice_token_indices.append([])
        
        slice_input_ids = torch.full((1, self.num_slices, self.slice_len), pad_id, dtype=torch.long)
        slice_attention_mask = torch.zeros((1, self.num_slices, self.slice_len), dtype=torch.float)
        valid_slices = 0
        
        slice_pos_mapping: List[Tuple[int, int]] = [(-1, -1)] * len(normalized)
        
        for s_idx, (slice_tokens, token_indices) in enumerate(zip(slices, slice_token_indices)):
            if not slice_tokens:
                continue
            
            ids = [bos_id]
            for j, t in enumerate(slice_tokens[:self.slice_len - 2]):
                ids.append(self.vocab.get(t, unk_id))
                if j < len(token_indices):
                    tok_idx = token_indices[j]
                    slice_pos_mapping[tok_idx] = (s_idx, j + 1)
            ids.append(eos_id)
            
            length = min(len(ids), self.slice_len)
            slice_input_ids[0, s_idx, :length] = torch.tensor(ids[:length], dtype=torch.long)
            slice_attention_mask[0, s_idx, :length] = 1.0
            valid_slices += 1
        
        slice_count = torch.tensor([max(1, valid_slices)], dtype=torch.long)
        
        vuln_features = self._extract_and_normalize_features(code, tokens)
        
        slice_vuln_features = torch.zeros((1, self.num_slices, VULN_DIM), dtype=torch.float)
        slice_rel_features = torch.zeros((1, self.num_slices, VULN_DIM), dtype=torch.float)
        
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "slice_input_ids": slice_input_ids,
            "slice_attention_mask": slice_attention_mask,
            "slice_count": slice_count,
            "vuln_features": vuln_features,
            "slice_vuln_features": slice_vuln_features,
            "slice_rel_features": slice_rel_features,
        }
        
        return inputs, tokens, token_lines, global_token_indices, slice_pos_mapping
    
    def _preprocess(self, code: str) -> Dict[str, torch.Tensor]:
        """Preprocess C code into model input tensors matching training format."""
        inputs, _, _, _, _ = self._preprocess_with_mapping(code)
        return inputs
    
    def _extract_and_normalize_features(self, code: str, tokens: List[str]) -> torch.Tensor:
        """Extract and normalize vulnerability features."""
        feats = extract_vuln_features(code, tokens)
        
        if self.feature_stats:
            feature_names = [
                "loc", "stmt_count", "dangerous_call_count",
                "dangerous_call_without_check_count", "dangerous_call_without_check_ratio",
                "pointer_deref_count", "pointer_deref_without_null_check_count",
                "pointer_deref_without_null_check_ratio", "array_access_count",
                "array_access_without_bounds_check_count", "array_access_without_bounds_check_ratio",
                "malloc_count", "malloc_without_free_count", "malloc_without_free_ratio",
                "free_count", "free_without_null_check_count", "free_without_null_check_ratio",
                "unchecked_return_value_count", "unchecked_return_value_ratio",
                "null_check_count", "bounds_check_count", "defense_ratio",
                "dangerous_call_density", "pointer_deref_density",
                "array_access_density", "null_check_density"
            ]
            
            for i, name in enumerate(feature_names):
                if i >= len(feats):
                    break
                stats = self.feature_stats.get(name, {})
                mean = stats.get("mean", 0.0)
                std = stats.get("std", 1.0)
                if std == 0:
                    std = 1.0
                feats[i] = (feats[i] - mean) / std
        
        return torch.from_numpy(feats).unsqueeze(0)

    def _get_confidence(self, score: float) -> str:
        if score > 0.8 or score < 0.2:
            return "high"
        elif score > 0.6 or score < 0.4:
            return "medium"
        return "low"
    
    def _compute_token_importance(
        self,
        attention: AttentionWeights,
        global_token_indices: List[int],
        slice_pos_mapping: List[Tuple[int, int]],
        num_tokens: int,
        w_global: float = 0.5,
        w_slice: float = 0.5,
    ) -> np.ndarray:
        """Combine attention layers into per-token importance scores."""
        global_alpha = attention.global_alpha[0].cpu().numpy()
        slice_token_alpha = attention.slice_token_alpha[0].cpu().numpy()
        slice_level_alpha = attention.slice_level_alpha[0].cpu().numpy()
        slice_seq_alpha = attention.slice_seq_alpha[0].cpu().numpy()
        
        importance = np.zeros(num_tokens, dtype=np.float32)
        
        for i, tok_idx in enumerate(global_token_indices):
            if tok_idx < num_tokens:
                global_pos = i + 1
                if global_pos < len(global_alpha):
                    importance[tok_idx] += w_global * global_alpha[global_pos]
        
        slice_combined = 0.5 * (slice_level_alpha + slice_seq_alpha)
        
        for tok_idx, (s_idx, slice_pos) in enumerate(slice_pos_mapping):
            if tok_idx >= num_tokens:
                break
            if s_idx >= 0 and slice_pos >= 0:
                if s_idx < len(slice_combined) and slice_pos < slice_token_alpha.shape[1]:
                    slice_weight = slice_combined[s_idx] * slice_token_alpha[s_idx, slice_pos]
                    importance[tok_idx] += w_slice * slice_weight
        
        total = importance.sum()
        if total > 0:
            importance = importance / total
        
        return importance
    
    def _aggregate_to_lines(
        self,
        importance: np.ndarray,
        token_lines: List[int],
        tokens: List[str],
        code: str,
    ) -> List[VulnerableLocation]:
        """Aggregate token importance to line-level scores."""
        line_scores: Dict[int, float] = {}
        line_tokens: Dict[int, List[str]] = {}
        
        for i, (imp, line) in enumerate(zip(importance, token_lines)):
            if line not in line_scores:
                line_scores[line] = 0.0
                line_tokens[line] = []
            line_scores[line] += imp
            if imp > 0.001 and i < len(tokens):
                line_tokens[line].append(tokens[i])
        
        if not line_scores:
            return []
        
        max_score = max(line_scores.values())
        if max_score == 0:
            return []
        
        code_lines = code.split('\n')
        
        locations = []
        for line, score in sorted(line_scores.items(), key=lambda x: -x[1]):
            norm_score = score / max_score
            snippet = code_lines[line - 1].strip() if line <= len(code_lines) else ""
            locations.append(VulnerableLocation(
                line=line,
                score=float(score),
                normalized_score=float(norm_score),
                code_snippet=snippet[:100],
                tokens=line_tokens.get(line, [])[:10],
            ))
        
        return locations

    @torch.inference_mode()
    def predict(self, code: str) -> PredictionResponse:
        """Predict using ensemble of models (average probabilities)."""
        inputs = self._preprocess(code)
        
        for k, v in inputs.items():
            inputs[k] = v.to(DEVICE)
        
        probs = []
        for model in self.models:
            logits = model(**inputs)
            prob = F.softmax(logits, dim=1)[0, 1].item()
            probs.append(prob)
        
        avg_prob = sum(probs) / len(probs)
        vulnerable = avg_prob >= self.threshold
        
        return PredictionResponse(
            vulnerable=vulnerable,
            score=round(avg_prob, 4),
            threshold=self.threshold,
            confidence=self._get_confidence(avg_prob),
        )
    
    @torch.inference_mode()
    def predict_with_localization(
        self, 
        code: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
        localize_margin: float = 0.05,
    ) -> LocalizationResult:
        """Predict vulnerability using ensemble and localize suspicious code lines."""
        inputs, tokens, token_lines, global_token_indices, slice_pos_mapping = self._preprocess_with_mapping(code)
        
        for k, v in inputs.items():
            inputs[k] = v.to(DEVICE)
        
        probs = []
        all_attentions = []
        for model in self.models:
            logits, attention = model.forward_with_attention(**inputs)
            prob = F.softmax(logits, dim=1)[0, 1].item()
            probs.append(prob)
            all_attentions.append(attention)
        
        avg_prob = sum(probs) / len(probs)
        vulnerable = avg_prob >= self.threshold
        
        prediction = PredictionResponse(
            vulnerable=vulnerable,
            score=round(avg_prob, 4),
            threshold=self.threshold,
            confidence=self._get_confidence(avg_prob),
        )
        
        highlights = []
        if vulnerable and avg_prob >= self.threshold + localize_margin:
            avg_global_alpha = sum(a.global_alpha for a in all_attentions) / len(all_attentions)
            avg_slice_token_alpha = sum(a.slice_token_alpha for a in all_attentions) / len(all_attentions)
            avg_slice_level_alpha = sum(a.slice_level_alpha for a in all_attentions) / len(all_attentions)
            avg_slice_seq_alpha = sum(a.slice_seq_alpha for a in all_attentions) / len(all_attentions)
            
            avg_attention = AttentionWeights(
                global_alpha=avg_global_alpha,
                slice_token_alpha=avg_slice_token_alpha,
                slice_level_alpha=avg_slice_level_alpha,
                slice_seq_alpha=avg_slice_seq_alpha,
            )
            
            importance = self._compute_token_importance(
                avg_attention, global_token_indices, slice_pos_mapping, len(tokens)
            )
            
            all_locations = self._aggregate_to_lines(importance, token_lines, tokens, code)
            
            highlights = [
                loc for loc in all_locations 
                if loc.normalized_score >= score_threshold
            ][:top_k]
            
            if not highlights and all_locations:
                highlights = [all_locations[0]]
        
        return LocalizationResult(prediction=prediction, highlights=highlights)


_model_wrapper: Optional[ModelWrapper] = None

def get_model_wrapper() -> ModelWrapper:
    global _model_wrapper
    if _model_wrapper is None:
        _model_wrapper = ModelWrapper()
    return _model_wrapper
