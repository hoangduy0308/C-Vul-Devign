# %% [markdown]
# # Word2Vec Training for Devign Vulnerability Detection
#
# Train Word2Vec embeddings on tokenized code corpus for use in BiGRU model.
#
# **Steps**:
# 1. Load vocab and tokenized sequences
# 2. Convert token IDs back to tokens
# 3. Train Word2Vec with skip-gram
# 4. Create embedding matrix for PyTorch
# 5. Save model and embedding matrix

# %%
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_environment() -> Tuple[Path, Path]:
    """Detect running environment and return appropriate paths."""
    kaggle_input = Path('/kaggle/input')
    kaggle_working = Path('/kaggle/working')
    
    if kaggle_input.exists():
        logger.info("Running on Kaggle")
        # Check possible dataset locations
        possible_paths = [
            kaggle_input / 'devign-final' / 'processed',
            kaggle_input / 'devign-final',
            kaggle_input / 'devign-processed',
            kaggle_working / 'processed',
        ]
        data_dir = None
        for p in possible_paths:
            if (p / 'vocab.json').exists():
                data_dir = p
                logger.info(f"Found data at: {data_dir}")
                break
        if data_dir is None:
            # Fallback - list available files for debugging
            logger.error("Could not find vocab.json. Available paths:")
            for p in kaggle_input.iterdir():
                logger.error(f"  {p}")
            data_dir = kaggle_input / 'devign-final'
        output_dir = kaggle_working
    else:
        logger.info("Running locally (Windows)")
        data_dir = Path('f:/Work/C Vul Devign/Dataset/devign_final')
        output_dir = data_dir
    
    return data_dir, output_dir


def load_vocab(vocab_path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load vocabulary and create reverse mapping."""
    logger.info(f"Loading vocab from {vocab_path}")
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        token_to_id = json.load(f)
    
    id_to_token = {v: k for k, v in token_to_id.items()}
    logger.info(f"Loaded vocab with {len(token_to_id)} tokens")
    
    return token_to_id, id_to_token


def load_tokenized_sequences(npz_path: Path, id_to_token: Dict[int, str]) -> List[List[str]]:
    """Load tokenized sequences and convert IDs back to tokens."""
    logger.info(f"Loading sequences from {npz_path}")
    
    data = np.load(npz_path)
    input_ids = data['input_ids']
    
    logger.info(f"Loaded {len(input_ids)} sequences, shape: {input_ids.shape}")
    
    token_sequences = []
    pad_id = 0
    
    for seq in input_ids:
        tokens = []
        for token_id in seq:
            if token_id == pad_id:
                continue
            token = id_to_token.get(token_id, '<UNK>')
            if token not in ['PAD', 'BOS', 'EOS']:
                tokens.append(token)
        if tokens:
            token_sequences.append(tokens)
    
    logger.info(f"Converted {len(token_sequences)} non-empty sequences")
    
    avg_len = sum(len(s) for s in token_sequences) / len(token_sequences) if token_sequences else 0
    logger.info(f"Average sequence length: {avg_len:.1f} tokens")
    
    return token_sequences


def train_word2vec(
    sentences: List[List[str]],
    vector_size: int = 128,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 20,
    seed: int = 42
):
    """Train Word2Vec model on token sequences."""
    try:
        from gensim.models import Word2Vec
    except ImportError:
        logger.error("gensim not installed. Run: pip install gensim")
        raise
    
    logger.info("Training Word2Vec model...")
    logger.info(f"  vector_size={vector_size}, window={window}, min_count={min_count}")
    logger.info(f"  epochs={epochs}, skip-gram=True, negative=10")
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,
        negative=10,
        epochs=epochs,
        workers=4,
        seed=seed
    )
    
    logger.info(f"Word2Vec vocabulary size: {len(model.wv)}")
    
    return model


def create_embedding_matrix(
    model,
    token_to_id: Dict[str, int],
    vocab_size: int = 30000,
    embed_dim: int = 128
) -> Tuple[np.ndarray, Dict]:
    """Create embedding matrix from Word2Vec model."""
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
    
    coverage = found_count / (vocab_size - len(special_tokens)) * 100
    
    stats = {
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'found_in_w2v': found_count,
        'oov_tokens': oov_count,
        'coverage_percent': coverage,
        'w2v_vocab_size': len(model.wv)
    }
    
    logger.info(f"Embedding matrix stats:")
    logger.info(f"  Found in Word2Vec: {found_count} ({coverage:.1f}%)")
    logger.info(f"  OOV tokens: {oov_count}")
    
    return embedding_matrix, stats


def print_similar_words(model, test_words: List[str], topn: int = 5):
    """Print similar words for key tokens."""
    logger.info("\nSimilar words for key tokens:")
    
    for word in test_words:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=topn)
            similar_str = ', '.join([f"{w}({s:.2f})" for w, s in similar])
            logger.info(f"  {word}: {similar_str}")
        else:
            logger.info(f"  {word}: <not in vocabulary>")


def save_outputs(
    model,
    embedding_matrix: np.ndarray,
    stats: Dict,
    output_dir: Path,
    config_path: Path,
    source_config_path: Path = None
):
    """Save Word2Vec model, embedding matrix, and update config."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / 'word2vec.model'
    model.save(str(model_path))
    logger.info(f"Saved Word2Vec model to {model_path}")
    
    matrix_path = output_dir / 'embedding_matrix.npy'
    np.save(matrix_path, embedding_matrix)
    logger.info(f"Saved embedding matrix to {matrix_path}")
    logger.info(f"  Shape: {embedding_matrix.shape}, dtype: {embedding_matrix.dtype}")
    
    # Load from source config (input dir) if exists, otherwise start fresh
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


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("Word2Vec Training for Devign Vulnerability Detection")
    logger.info("=" * 60)
    
    data_dir, output_dir = detect_environment()
    
    vocab_path = data_dir / 'vocab.json'
    train_npz_path = data_dir / 'train.npz'
    source_config_path = data_dir / 'config.json'  # Read from input (read-only)
    config_path = output_dir / 'config.json'  # Save to output_dir (writable)
    
    for path in [vocab_path, train_npz_path]:
        if not path.exists():
            logger.error(f"Required file not found: {path}")
            return
    
    token_to_id, id_to_token = load_vocab(vocab_path)
    
    token_sequences = load_tokenized_sequences(train_npz_path, id_to_token)
    
    val_npz_path = data_dir / 'val.npz'
    if val_npz_path.exists():
        val_sequences = load_tokenized_sequences(val_npz_path, id_to_token)
        token_sequences.extend(val_sequences)
        logger.info(f"Added validation sequences. Total: {len(token_sequences)}")
    
    model = train_word2vec(
        sentences=token_sequences,
        vector_size=128,
        window=5,
        min_count=2,
        epochs=20,
        seed=42
    )
    
    test_words = ['malloc', 'memcpy', 'free', 'if', 'for', 'return', 'NULL', 'strlen', 'buffer', 'size']
    print_similar_words(model, test_words)
    
    embedding_matrix, stats = create_embedding_matrix(
        model=model,
        token_to_id=token_to_id,
        vocab_size=30000,
        embed_dim=128
    )
    
    save_outputs(model, embedding_matrix, stats, output_dir, config_path, source_config_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info(f"  Embedding matrix: {embedding_matrix.shape}")
    logger.info(f"  Coverage: {stats['coverage_percent']:.1f}%")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
