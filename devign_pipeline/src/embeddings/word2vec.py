"""
Word2Vec training utilities for vulnerability detection embeddings.

This module provides functions for training Word2Vec embeddings on tokenized
code sequences and creating embedding matrices for PyTorch models.

Usage:
    from src.embeddings.word2vec import train_word2vec, create_embedding_matrix
    
    model = train_word2vec(sentences, config={'vector_size': 128})
    matrix, stats = create_embedding_matrix(model, vocab)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


# Default Word2Vec configuration
DEFAULT_W2V_CONFIG = {
    'vector_size': 128,
    'window': 5,
    'min_count': 2,  # Filter noise from single-occurrence tokens (they get poor vectors)
    'epochs': 20,
    'sg': 1,  # skip-gram (1) vs CBOW (0)
    'negative': 10,
    'workers': 4,
    'seed': 42,
}


def calculate_weighted_coverage(
    model: Any,
    token_sequences: List[List[str]],
    vocab: Dict[str, int]
) -> Dict[str, Any]:
    """
    Calculate occurrence-weighted Word2Vec coverage.
    
    This measures what percentage of actual token occurrences in the corpus
    have learned Word2Vec vectors, giving a more accurate picture than
    simple vocabulary coverage.
    
    Args:
        model: Trained gensim Word2Vec model
        token_sequences: List of token sequences from the corpus
        vocab: Vocabulary dict (token -> id)
    
    Returns:
        Dict with:
        - total_occurrences: Total token count
        - covered_occurrences: Tokens with learned vectors  
        - weighted_coverage: covered/total as percentage
        - uncovered_tokens: List of (token, count) for OOV tokens sorted by frequency
    """
    from collections import Counter
    
    token_counts: Counter = Counter()
    for seq in token_sequences:
        token_counts.update(seq)
    
    special_tokens = {'PAD', 'UNK', 'BOS', 'EOS', 'SEP'}
    
    total_occurrences = 0
    covered_occurrences = 0
    uncovered: Dict[str, int] = {}
    
    for token, count in token_counts.items():
        if token in special_tokens:
            continue
        total_occurrences += count
        if token in model.wv:
            covered_occurrences += count
        else:
            uncovered[token] = count
    
    weighted_coverage = (covered_occurrences / total_occurrences * 100) if total_occurrences > 0 else 0.0
    
    uncovered_sorted = sorted(uncovered.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'total_occurrences': total_occurrences,
        'covered_occurrences': covered_occurrences,
        'weighted_coverage': weighted_coverage,
        'uncovered_tokens': uncovered_sorted
    }


def train_word2vec(
    sentences: List[List[str]],
    config: Optional[Dict[str, Any]] = None,
    vocab: Optional[Dict[str, int]] = None,
) -> Any:
    """Train Word2Vec model on token sequences.
    
    Args:
        sentences: List of tokenized sequences (list of token strings).
            Each sequence is a list of tokens representing one code sample.
        config: Word2Vec configuration dict. Uses DEFAULT_W2V_CONFIG if None.
            Keys: vector_size, window, min_count, epochs, sg, negative, workers, seed
        vocab: Optional vocabulary dict (token -> id). Used for logging only.
    
    Returns:
        Trained gensim Word2Vec model
        
    Raises:
        ImportError: If gensim is not installed
        
    Example:
        >>> sentences = [['malloc', 'ptr', 'free'], ['memcpy', 'buf', 'len']]
        >>> model = train_word2vec(sentences, config={'vector_size': 64})
    """
    try:
        from gensim.models import Word2Vec
    except ImportError:
        logger.error("gensim not installed. Run: pip install gensim")
        raise
    
    cfg = {**DEFAULT_W2V_CONFIG, **(config or {})}
    
    logger.info("Training Word2Vec model...")
    logger.info(f"  vector_size={cfg['vector_size']}, window={cfg['window']}, min_count={cfg['min_count']}")
    logger.info(f"  epochs={cfg['epochs']}, skip-gram={cfg['sg']==1}, negative={cfg['negative']}")
    logger.info(f"  Training on {len(sentences)} sequences")
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=cfg['vector_size'],
        window=cfg['window'],
        min_count=cfg['min_count'],
        sg=cfg['sg'],
        negative=cfg['negative'],
        epochs=cfg['epochs'],
        workers=cfg['workers'],
        seed=cfg['seed']
    )
    
    logger.info(f"Word2Vec vocabulary size: {len(model.wv)}")
    if vocab:
        logger.info(f"Target vocab size: {len(vocab)}")
    
    return model


def create_embedding_matrix(
    model: Any,
    token_to_id: Dict[str, int],
    vocab_size: Optional[int] = None,
    embed_dim: Optional[int] = None,
    token_sequences: Optional[List[List[str]]] = None
) -> Tuple[np.ndarray, Dict]:
    """Create embedding matrix from Word2Vec model.
    
    Args:
        model: Trained gensim Word2Vec model
        token_to_id: Vocabulary dict (token -> id)
        vocab_size: Target vocab size. Defaults to len(token_to_id) or 30000.
        embed_dim: Embedding dimension. Defaults to model.wv.vector_size.
        token_sequences: Optional token sequences for weighted coverage calculation.
            If provided, stats will include weighted_coverage metrics.
    
    Returns:
        Tuple of (embedding_matrix, stats_dict) where:
        - embedding_matrix: np.ndarray of shape (vocab_size, embed_dim)
        - stats_dict: Dict with coverage statistics (includes weighted_coverage if token_sequences provided)
        
    Example:
        >>> matrix, stats = create_embedding_matrix(model, vocab)
        >>> print(f"Coverage: {stats['coverage_percent']:.1f}%")
    """
    if vocab_size is None:
        vocab_size = min(len(token_to_id), 30000)
    if embed_dim is None:
        embed_dim = model.wv.vector_size
    
    logger.info(f"Creating embedding matrix: ({vocab_size}, {embed_dim})")
    
    embedding_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    
    all_vectors = model.wv.vectors
    mean_vector = np.mean(all_vectors, axis=0)
    std_vector = np.std(all_vectors, axis=0).mean()
    
    found_count = 0
    oov_count = 0
    special_tokens = {'PAD': 0, 'UNK': 1, 'BOS': 2, 'EOS': 3, 'SEP': 4}
    
    for token, idx in token_to_id.items():
        if idx >= vocab_size:
            continue
        
        if token == 'PAD':
            embedding_matrix[idx] = np.zeros(embed_dim)
        elif token == 'UNK':
            embedding_matrix[idx] = mean_vector
        elif token in special_tokens:
            np.random.seed(42 + idx)
            embedding_matrix[idx] = np.random.normal(0, std_vector * 0.1, embed_dim)
        elif token in model.wv:
            embedding_matrix[idx] = model.wv[token]
            found_count += 1
        else:
            embedding_matrix[idx] = mean_vector + np.random.normal(0, std_vector * 0.1, embed_dim)
            oov_count += 1
    
    coverage = found_count / (vocab_size - len(special_tokens)) * 100 if vocab_size > len(special_tokens) else 0
    
    stats = {
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'found_in_w2v': found_count,
        'oov_tokens': oov_count,
        'coverage_percent': coverage,
        'w2v_vocab_size': len(model.wv)
    }
    
    if token_sequences is not None:
        weighted_stats = calculate_weighted_coverage(model, token_sequences, token_to_id)
        stats['weighted_coverage'] = weighted_stats['weighted_coverage']
        stats['total_occurrences'] = weighted_stats['total_occurrences']
        stats['covered_occurrences'] = weighted_stats['covered_occurrences']
        stats['top_uncovered_tokens'] = weighted_stats['uncovered_tokens'][:20]
        logger.info(f"  Weighted coverage: {weighted_stats['weighted_coverage']:.1f}% "
                   f"({weighted_stats['covered_occurrences']}/{weighted_stats['total_occurrences']} occurrences)")
    
    logger.info(f"Embedding matrix stats:")
    logger.info(f"  Found in Word2Vec: {found_count} ({coverage:.1f}%)")
    logger.info(f"  OOV tokens: {oov_count}")
    
    return embedding_matrix, stats


def tokens_from_input_ids(
    input_ids: np.ndarray,
    id_to_token: Dict[int, str],
    skip_special: bool = True
) -> List[List[str]]:
    """Convert input_ids array to token sequences for Word2Vec training.
    
    Args:
        input_ids: Numpy array of shape (n_samples, seq_len) with token IDs
        id_to_token: Reverse vocabulary mapping (id -> token)
        skip_special: Whether to skip PAD, BOS, EOS tokens (default True)
    
    Returns:
        List of token sequences (list of strings)
        
    Example:
        >>> id_to_token = {0: 'PAD', 1: 'UNK', 5: 'malloc', 6: 'free'}
        >>> input_ids = np.array([[5, 6, 0, 0]])
        >>> sequences = tokens_from_input_ids(input_ids, id_to_token)
        >>> # sequences = [['malloc', 'free']]
    """
    token_sequences = []
    pad_id = 0
    special_tokens = {'PAD', 'BOS', 'EOS'} if skip_special else set()
    
    for seq in input_ids:
        tokens = []
        for token_id in seq:
            if token_id == pad_id:
                continue
            token = id_to_token.get(int(token_id), '<UNK>')
            if token not in special_tokens:
                tokens.append(token)
        if tokens:
            token_sequences.append(tokens)
    
    return token_sequences


def save_word2vec_outputs(
    model: Any,
    embedding_matrix: np.ndarray,
    stats: Dict,
    output_dir: Path,
    save_model: bool = True,
    update_config: bool = True,
    config_path: Optional[Path] = None,
    source_config_path: Optional[Path] = None
) -> Dict[str, Path]:
    """Save Word2Vec model, embedding matrix, and optionally update config.
    
    Args:
        model: Trained gensim Word2Vec model
        embedding_matrix: Numpy embedding matrix
        stats: Statistics dict from create_embedding_matrix
        output_dir: Directory to save outputs
        save_model: Whether to save the gensim model file
        update_config: Whether to update config.json with Word2Vec info
        config_path: Path to config.json to update (defaults to output_dir/config.json)
        source_config_path: Path to source config to copy from (for Kaggle read-only input)
    
    Returns:
        Dict with paths to saved files ('model', 'embedding_matrix', 'config')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    if save_model:
        model_path = output_dir / 'word2vec.model'
        model.save(str(model_path))
        logger.info(f"Saved Word2Vec model to {model_path}")
        saved_files['model'] = model_path
    
    matrix_path = output_dir / 'embedding_matrix.npy'
    np.save(matrix_path, embedding_matrix)
    logger.info(f"Saved embedding matrix to {matrix_path}")
    logger.info(f"  Shape: {embedding_matrix.shape}, dtype: {embedding_matrix.dtype}")
    saved_files['embedding_matrix'] = matrix_path
    
    if update_config:
        if config_path is None:
            config_path = output_dir / 'config.json'
        
        config = {}
        if source_config_path and source_config_path.exists():
            with open(source_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        elif config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        
        config['word2vec'] = {
            'vector_size': stats['embed_dim'],
            'vocab_size': stats['vocab_size'],
            'w2v_vocab_size': stats['w2v_vocab_size'],
            'coverage_percent': stats['coverage_percent'],
            'found_in_w2v': stats['found_in_w2v'],
            'oov_tokens': stats['oov_tokens']
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Updated config at {config_path}")
        saved_files['config'] = config_path
    
    return saved_files


def print_similar_words(
    model: Any, 
    test_words: Optional[List[str]] = None, 
    topn: int = 5,
    include_semantic_buckets: bool = True
) -> None:
    """Print similar words for key tokens (for debugging/analysis).
    
    Args:
        model: Trained gensim Word2Vec model
        test_words: List of words to find similar words for. If None, uses defaults.
        topn: Number of similar words to show
        include_semantic_buckets: Whether to also test semantic bucket tokens (BUF_0, LEN_0, etc.)
    """
    default_words = ['malloc', 'free', 'memcpy', 'NULL', 'buffer', 'size', 'len', 'ptr']
    semantic_bucket_words = ['BUF_0', 'LEN_0', 'PTR_0', 'IDX_0', 'ERR_0']
    
    words_to_test = test_words if test_words is not None else default_words
    if include_semantic_buckets:
        words_to_test = list(words_to_test) + semantic_bucket_words
    
    logger.info("\nSimilar words for key tokens:")
    
    for word in words_to_test:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=topn)
            similar_str = ', '.join([f"{w}({s:.2f})" for w, s in similar])
            logger.info(f"  {word}: {similar_str}")
        else:
            logger.info(f"  {word}: <not in vocabulary>")
