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
#
# **Usage**:
# - Standalone: `python 01b_train_word2vec.py`
# - As import: Functions are in `src.embeddings.word2vec`

# %%
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add pipeline root to path for imports
NOTEBOOK_DIR = Path(__file__).parent if '__file__' in dir() else Path.cwd()
PIPELINE_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == 'notebooks' else NOTEBOOK_DIR
sys.path.insert(0, str(PIPELINE_ROOT))

# Import from src.embeddings
from src.embeddings.word2vec import (
    train_word2vec,
    create_embedding_matrix,
    save_word2vec_outputs,
    tokens_from_input_ids,
    print_similar_words,
    DEFAULT_W2V_CONFIG,
)


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
            logger.error("Could not find vocab.json. Available paths:")
            for p in kaggle_input.iterdir():
                logger.error(f"  {p}")
            data_dir = kaggle_input / 'devign-final'
        output_dir = kaggle_working
    else:
        logger.info("Running locally")
        data_dir = PIPELINE_ROOT.parent / 'output' / 'processed'
        output_dir = data_dir.parent
    
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
    
    token_sequences = tokens_from_input_ids(input_ids, id_to_token, skip_special=True)
    
    logger.info(f"Converted {len(token_sequences)} non-empty sequences")
    
    avg_len = sum(len(s) for s in token_sequences) / len(token_sequences) if token_sequences else 0
    logger.info(f"Average sequence length: {avg_len:.1f} tokens")
    
    return token_sequences


def main():
    """Main training function for standalone execution."""
    logger.info("=" * 60)
    logger.info("Word2Vec Training for Devign Vulnerability Detection")
    logger.info("=" * 60)
    
    data_dir, output_dir = detect_environment()
    
    vocab_path = data_dir / 'vocab.json'
    train_npz_path = data_dir / 'train.npz'
    source_config_path = data_dir / 'config.json'
    config_path = output_dir / 'config.json'
    
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
        config=DEFAULT_W2V_CONFIG,
        vocab=token_to_id
    )
    
    test_words = ['malloc', 'memcpy', 'free', 'if', 'for', 'return', 'NULL', 'strlen', 'buffer', 'size']
    print_similar_words(model, test_words)
    
    embedding_matrix, stats = create_embedding_matrix(
        model=model,
        token_to_id=token_to_id,
        vocab_size=len(token_to_id),
        embed_dim=DEFAULT_W2V_CONFIG['vector_size']
    )
    
    save_word2vec_outputs(
        model=model,
        embedding_matrix=embedding_matrix,
        stats=stats,
        output_dir=output_dir,
        save_model=True,
        update_config=True,
        config_path=config_path,
        source_config_path=source_config_path
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info(f"  Embedding matrix: {embedding_matrix.shape}")
    logger.info(f"  Coverage: {stats['coverage_percent']:.1f}%")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
