import javalang
import pandas as pd
import json
import re

class CodeTokenizerJava:
    SPECIAL_NUMBERS = {
        "0", "1", "-1",
        "20", "21", "22", "23", "25", "53", "80", "110", "119", "123", "137", "143", "161",
        "443", "465", "587", "993", "995", "1433", "1521", "3306", "3389", "5432", "8080"
    }

    def __init__(self):
        self.token_sequences = []

    # --- Nhận diện số ---
    @classmethod
    def map_num_bit(cls, tok):
        if tok in cls.SPECIAL_NUMBERS:
            return tok
        if re.fullmatch(r"-?\d+\.\d+", tok):  # float/double
            val = float(tok)
            if -90 <= val <= 90:
                return "<gps_lat>"
            elif -180 <= val <= 180:
                return "<gps_lon>"
            return "<num_float>"
        if re.fullmatch(r"-?\d+", tok):  # integer
            val = abs(int(tok))
            if val < 2**8:
                return "<num_8bit>"
            elif val < 2**16:
                return "<num_16bit>"
            elif val < 2**32:
                return "<num_32bit>"
            return "<num_large>"
        return tok

    # --- Nhận diện string literal ---
    @staticmethod
    def map_string_literal(lit):
        val = lit.strip('"\'')
        upper_val = val.upper()

        if re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|FROM|WHERE|JOIN|GROUP BY|ORDER BY|LIMIT|UNION|HAVING)\b", upper_val):
            return "<sql_str>"
        if re.search(r"[.*+?|^$\\\[\]{}()]", val) and len(val) > 3 and not re.match(r"^https?://", val):
            return "<regex_str>"
        if re.match(r"^(https?|ftp|file)://", val) or re.match(r".+\.(com|net|org|io|gov|edu)(/|$)", val, re.IGNORECASE):
            return "<url_str>"
        if re.search(r"[\\/]", val) or re.match(r"([A-Za-z]:)?[\\/]", val):
            return "<path_str>"
        if re.search(r"(pass(word)?|secret|token|key)", val, re.IGNORECASE):
            return "<cred_str>"
        if re.match(r"[^@]+@[^@]+\.[^@]+", val):
            return "<email_str>"
        if re.match(r"\b\d{1,3}(\.\d{1,3}){3}\b", val) or re.match(r"([0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}", val):
            return "<ip_str>"
        if upper_val in {"GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"}:
            return "<http_method>"
        if upper_val.startswith(("ERROR", "WARN", "INFO", "DEBUG")):
            return "<log_msg>"
        if re.match(r".+\.[a-zA-Z0-9]{1,5}$", val) and "/" not in val and "\\" not in val:
            return "<file_str>"
        return val

    # --- Tokenize bằng javalang ---
    def tokenize(self, java_code: str):
        try:
            tokens = list(javalang.tokenizer.tokenize(java_code))
        except javalang.tokenizer.LexerError:
            tokens = []

        processed = []
        for t in tokens:
            if isinstance(t, javalang.tokenizer.String):
                processed.append(self.map_string_literal(t.value))
            elif isinstance(t, javalang.tokenizer.Integer) or isinstance(t, javalang.tokenizer.DecimalInteger):
                processed.append(self.map_num_bit(t.value))
            elif isinstance(t, javalang.tokenizer.FloatingPoint):
                processed.append(self.map_num_bit(t.value))
            else:
                processed.append(t.value)

        self.token_sequences.append(processed)
        return processed

    def get_token_sequences(self):
        return self.token_sequences

# ==== Example usage ====

def main(input_csv, output_csv):
    df = pd.read_csv(input_csv, encoding="utf-8")
    tokenizer = CodeTokenizerJava()

    required_cols = [
        "label", "cve_id", "method_name", "code_snippet", "filename", "commit_hash",
        "programming_language", "cwe_id", "repo_url", "author_date", "commit_msg"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Thiếu cột bắt buộc: {col}")

    token_results = []
    for _, row in df.iterrows():
        tokens = tokenizer.tokenize(str(row["code_snippet"]))
        token_results.append(tokens)

    df["tokens"] = [json.dumps(toks, ensure_ascii=False) for toks in token_results]
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[+] Tokenized dataset saved to {output_csv}")


if __name__ == "__main__":
    main("cleaned_filtered_methods.csv", "dataset_tokenized.csv")
