import re
import json
from collections import Counter
from pathlib import Path
from typing import List, Dict

class VocabularyManagerJava:
    """
    Vocab manager cho Java code: chỉ giữ token xuất hiện >= N lần,
    map số về <num>, còn lại (token hiếm) về <unk>.
    Hỗ trợ load/save, encode token, và reserved tokens chuẩn DL.
    """

    RESERVED = {
        "<pad>": 0,
        "<unk>": 1,
        "<num>": 2,
        "<sos>": 3,
        "<eos>": 4
    }

    def __init__(self, N: int = 3, max_vocab: int = 30000):
        self.N = N
        self.max_vocab = max_vocab
        self.token_to_idx: Dict[str, int] = dict(self.RESERVED)
        self.idx_to_token: Dict[int, str] = {v: k for k, v in self.RESERVED.items()}
        self.freq: Counter = Counter()
        self._is_built = False

    @staticmethod
    def _map_num(tok: str) -> str:
        return "<num>" if re.fullmatch(r"\d+", tok) else tok

    def count_token_lists(self, token_lists: List[List[str]]) -> None:
        for tokens in token_lists:
            for t in tokens:
                self.freq[self._map_num(t)] += 1

    def build_vocab(self) -> None:
        # Xây lại vocab: reserved + các token đủ tần suất (>=N)
        self.token_to_idx = dict(self.RESERVED)
        idx = max(self.RESERVED.values()) + 1
        for tok, cnt in self.freq.most_common():
            if tok in self.token_to_idx: continue
            if cnt >= self.N and idx < self.max_vocab:
                self.token_to_idx[tok] = idx
                idx += 1
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
        self._is_built = True

    def encode(self, tokens: List[str]) -> List[int]:
        mapped = [self._map_num(t) for t in tokens]
        unk = self.token_to_idx["<unk>"]
        return [self.token_to_idx.get(t, unk) for t in mapped]

    def decode(self, indices):
        return [self.idx_to_token.get(idx, "<unk>") for idx in indices]

    def save_vocab(self, path="vocabUpdate.json"):
        used_idx = set(self.token_to_idx.values())
        waiting = {f"token{i}": i for i in range(len(self.token_to_idx), self.max_vocab) if i not in used_idx}
        data = {
            "TOKEN": self.token_to_idx,
            "TOKEN_WAITING": waiting
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_vocab(self, path: str = "vocabUpdate.json") -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.token_to_idx = data.get("TOKEN", {})
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
        self._is_built = True

    def __len__(self) -> int:
        return len(self.token_to_idx)

    def get_index(self, token: str) -> int:
        return self.token_to_idx.get(token, self.token_to_idx["<unk>"])

    def get_token(self, index: int) -> str:
        return self.idx_to_token.get(index, "<unk>")
# import json

# # Đọc tất cả tokens từ file jsonl
# token_lists = []
# with open("filtered_labeled_slices.jsonl", "r", encoding="utf-8") as f:
#     for line in f:
#         data = json.loads(line)
#         tokens = data.get("tokens", [])
#         token_lists.append(tokens)

# # Build vocab
# vocab = VocabularyManagerJava(N=4, max_vocab=30000)
# vocab.count_token_lists(token_lists)
# vocab.build_vocab()
# vocab.save_vocab("vocabUpdate.json")

# print(f"Số lượng token trong vocab: {len(vocab)}")


# === Example usage ===
if __name__ == "__main__":
    import pandas as pd
    from CodeTokenizerJava import CodeTokenizerJava

    df = pd.read_csv("cleaned_filtered_methods.csv")
    tokenizer = CodeTokenizerJava()
    token_lists = [tokenizer.tokenize(str(code)) for code in df["code_snippet"]]

    vocab = VocabularyManagerJava(N=3, max_vocab=30000)
    vocab.count_token_lists(token_lists)
    vocab.build_vocab()
    vocab.save_vocab("vocabUpdate2.json")
    print("Số lượng token trong vocab:", len(vocab))

    # Encode lại từng đoạn code
    df["token_indices"] = [vocab.encode(tokens) for tokens in token_lists]
    df.to_csv("final_tokenized_methods.csv", index=False)
    print("Đã lưu final_tokenized_methods.csv")