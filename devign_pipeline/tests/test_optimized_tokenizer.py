"""
Tests for OptimizedHybridTokenizer

Run with: pytest tests/test_optimized_tokenizer.py -v
"""

import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenization.optimized_tokenizer import (
    OptimizedHybridTokenizer,
    build_optimized_vocab,
    vectorize_optimized,
    API_FAMILIES,
    DEFENSE_FAMILIES,
    SEMANTIC_BUCKETS,
    UNIVERSAL_DANGEROUS,
)


class TestOptimizedHybridTokenizer:
    """Test cases for OptimizedHybridTokenizer."""
    
    @pytest.fixture
    def tokenizer(self):
        return OptimizedHybridTokenizer()
    
    @pytest.fixture
    def sample_code(self):
        return '''
        void process_request(char *user_input, int size) {
            char buf[256];
            int len = strlen(user_input);
            
            if (len > 0 && len < 256) {
                strcpy(buf, user_input);
                printf("Input: %s", buf);
            }
            
            char *ptr = malloc(1024);
            if (ptr == NULL) {
                return -1;
            }
            
            memcpy(ptr, buf, len);
            ptr[len] = 0;
            free(ptr);
            return 0;
        }
        '''
    
    @pytest.fixture
    def ffmpeg_code(self):
        return '''
        AVFrame *frame = av_frame_alloc();
        uint8_t *buffer = av_malloc(size);
        if (!buffer) {
            av_frame_free(&frame);
            return AVERROR(ENOMEM);
        }
        memcpy(buffer, src_data, len);
        av_buffer_unref(&frame->buf[0]);
        av_free(buffer);
        '''
    
    def test_tokenize_basic(self, tokenizer, sample_code):
        """Test basic tokenization."""
        tokens = tokenizer.tokenize(sample_code)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
    
    def test_api_family_mapping(self, tokenizer, sample_code):
        """Test that dangerous APIs are properly tokenized."""
        tokens = tokenizer.tokenize(sample_code)
        
        # Universal dangerous APIs should be kept as-is
        assert 'malloc' in tokens
        assert 'free' in tokens
        assert 'memcpy' in tokens
        assert 'strcpy' in tokens
        assert 'strlen' in tokens
    
    def test_ffmpeg_api_families(self, tokenizer, ffmpeg_code):
        """Test that FFmpeg APIs are mapped to families."""
        tokens = tokenizer.tokenize(ffmpeg_code)
        
        # FFmpeg-specific APIs should be mapped to families
        assert 'API_ALLOC' in tokens  # av_malloc -> API_ALLOC
        assert 'API_FREE' in tokens   # av_free -> API_FREE
        assert 'API_UNREF' in tokens  # av_buffer_unref -> API_UNREF
        
        # Universal APIs should still be kept
        assert 'memcpy' in tokens
    
    def test_semantic_buckets(self, tokenizer, sample_code):
        """Test that identifiers are mapped to semantic buckets."""
        tokens = tokenizer.tokenize(sample_code)
        
        # Check for semantic bucket tokens
        bucket_tokens = [t for t in tokens if '_' in t and t.split('_')[0] in SEMANTIC_BUCKETS]
        assert len(bucket_tokens) > 0
        
        # Specific mappings
        var_mapping = tokenizer.var_to_canonical
        
        # 'buf' should map to BUF_*
        assert any(v.startswith('BUF_') for v in var_mapping.values())
        
        # 'len' should map to LEN_*
        assert any(v.startswith('LEN_') for v in var_mapping.values())
        
        # 'ptr' should map to PTR_*
        assert any(v.startswith('PTR_') for v in var_mapping.values())
    
    def test_c_keywords_preserved(self, tokenizer, sample_code):
        """Test that C keywords are preserved."""
        tokens = tokenizer.tokenize(sample_code)
        
        assert 'void' in tokens
        assert 'char' in tokens
        assert 'int' in tokens
        assert 'if' in tokens
        assert 'return' in tokens
        assert 'NULL' in tokens
    
    def test_number_handling(self, tokenizer):
        """Test smart number handling."""
        code = "int x = 0; int y = 256; int z = 12345;"
        tokens = tokenizer.tokenize(code)
        
        # Small integers preserved
        assert '0' in tokens
        
        # Powers of 2 preserved
        assert '256' in tokens
        
        # Large numbers normalized
        assert 'NUM' in tokens
    
    def test_negative_one_handling(self, tokenizer):
        """Test -1 is tokenized as NEG_1."""
        code = "return -1;"
        tokens = tokenizer.tokenize(code)
        
        assert 'NEG_1' in tokens
    
    def test_function_normalization(self, tokenizer):
        """Test that non-API functions are normalized to FUNC."""
        code = "my_custom_function(arg1, arg2);"
        tokens = tokenizer.tokenize(code)
        
        assert 'FUNC' in tokens
    
    def test_string_char_literals(self, tokenizer):
        """Test string and char literals are normalized."""
        code = 'char *s = "hello"; char c = \'x\';'
        tokens = tokenizer.tokenize(code)
        
        assert 'STR' in tokens
        assert 'CHAR' in tokens
    
    def test_operators_preserved(self, tokenizer, sample_code):
        """Test that operators are preserved."""
        tokens = tokenizer.tokenize(sample_code)
        
        assert '=' in tokens
        assert '==' in tokens
        assert '>' in tokens
        assert '<' in tokens
        assert '&&' in tokens
    
    def test_variable_mapping_reset(self, tokenizer):
        """Test that variable mappings reset between tokenizations."""
        code1 = "int buf = 1;"
        code2 = "int buf = 2;"
        
        tokenizer.tokenize(code1)
        mapping1 = tokenizer.var_to_canonical.copy()
        
        tokenizer.tokenize(code2)
        mapping2 = tokenizer.var_to_canonical.copy()
        
        # Both should have same mapping for 'buf'
        assert 'buf' in mapping1
        assert 'buf' in mapping2
        assert mapping1['buf'] == mapping2['buf']


class TestBuildOptimizedVocab:
    """Test cases for vocabulary building."""
    
    def test_build_vocab_basic(self):
        """Test basic vocabulary building."""
        tokens_list = [
            ['malloc', 'BUF_0', 'LEN_0', 'free'],
            ['memcpy', 'BUF_1', 'PTR_0', 'return'],
        ]
        
        vocab, debug = build_optimized_vocab(tokens_list, min_freq=1, max_size=1000)
        
        assert isinstance(vocab, dict)
        assert len(vocab) > 0
        assert 'PAD' in vocab
        assert 'UNK' in vocab
    
    def test_vocab_includes_api_families(self):
        """Test that API families are in vocab."""
        vocab, _ = build_optimized_vocab(None, max_size=2000)
        
        for family in API_FAMILIES.keys():
            assert family in vocab, f"{family} should be in vocab"
    
    def test_vocab_includes_semantic_buckets(self):
        """Test that semantic bucket tokens are in vocab."""
        vocab, _ = build_optimized_vocab(None, max_size=2000)
        
        # Check some bucket tokens exist
        assert 'BUF_0' in vocab
        assert 'LEN_0' in vocab
        assert 'PTR_0' in vocab
        assert 'VAR_0' in vocab


class TestVectorizeOptimized:
    """Test cases for vectorization."""
    
    @pytest.fixture
    def vocab(self):
        vocab, _ = build_optimized_vocab(None, max_size=2000)
        return vocab
    
    def test_vectorize_basic(self, vocab):
        """Test basic vectorization."""
        tokens = ['malloc', 'BUF_0', '(', 'LEN_0', ')', ';']
        
        input_ids, attention_mask, unk_positions = vectorize_optimized(
            tokens, vocab, max_len=32
        )
        
        assert len(input_ids) == 32
        assert len(attention_mask) == 32
        assert sum(attention_mask) == len(tokens)
    
    def test_vectorize_truncation(self, vocab):
        """Test truncation for long sequences."""
        tokens = ['int'] * 600  # Longer than max_len
        
        input_ids, attention_mask, _ = vectorize_optimized(
            tokens, vocab, max_len=512
        )
        
        assert len(input_ids) == 512
        assert len(attention_mask) == 512
    
    def test_vectorize_head_tail(self, vocab):
        """Test head_tail truncation strategy."""
        tokens = ['HEAD'] * 200 + ['MIDDLE'] * 200 + ['TAIL'] * 200
        
        input_ids, attention_mask, _ = vectorize_optimized(
            tokens, vocab, max_len=512,
            truncation_strategy='head_tail',
            head_tokens=192,
            tail_tokens=319
        )
        
        assert len(input_ids) == 512


class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_full_pipeline(self):
        """Test complete tokenize -> vocab -> vectorize pipeline."""
        code = '''
        void vulnerable_func(char *input) {
            char buffer[100];
            strcpy(buffer, input);
            printf("%s", buffer);
        }
        '''
        
        # Tokenize
        tokenizer = OptimizedHybridTokenizer()
        tokens = tokenizer.tokenize(code)
        
        assert len(tokens) > 0
        
        # Build vocab
        vocab, debug = build_optimized_vocab([tokens], min_freq=1, max_size=2000)
        
        assert debug['vocab_size'] > 0
        
        # Vectorize
        input_ids, attention_mask, unk_positions = vectorize_optimized(
            tokens, vocab, max_len=512
        )
        
        assert len(input_ids) == 512
        assert sum(attention_mask) == len(tokens)
        
        # UNK rate should be low
        unk_rate = len(unk_positions) / len(tokens) if tokens else 0
        assert unk_rate < 0.1, f"UNK rate too high: {unk_rate}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
