#!/usr/bin/env python3
"""
Inference script for CodeBERT vulnerability detection.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import RobertaTokenizer

from models.codebert_classifier import CodeBERTClassifier


class VulnerabilityPredictor:
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint["config"]

        self.model = CodeBERTClassifier(
            model_name=config["model"]["name"],
            head_type=config["model"]["head_type"],
            num_labels=config["model"]["num_labels"],
            dropout=config["model"]["dropout"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = self.model.tokenizer
        self.max_length = config["data"]["max_length"]

    def predict(
        self,
        code: Union[str, List[str]],
        return_probs: bool = True,
    ) -> Dict:
        if isinstance(code, str):
            code = [code]
            single_input = True
        else:
            single_input = False

        inputs = self.tokenizer(
            code,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)

        predictions = predictions.cpu().numpy()
        probs = probs.cpu().numpy()

        results = []
        for i in range(len(code)):
            result = {
                "prediction": int(predictions[i]),
                "label": "vulnerable" if predictions[i] == 1 else "safe",
            }
            if return_probs:
                result["probability"] = {
                    "safe": float(probs[i][0]),
                    "vulnerable": float(probs[i][1]),
                }
                result["confidence"] = float(max(probs[i]))
            results.append(result)

        if single_input:
            return results[0]
        return results

    def predict_file(self, file_path: str) -> Dict:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        return self.predict(code)

    def predict_batch(
        self,
        codes: List[str],
        batch_size: int = 32,
    ) -> List[Dict]:
        all_results = []

        for i in range(0, len(codes), batch_size):
            batch = codes[i : i + batch_size]
            results = self.predict(batch)
            all_results.extend(results)

        return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Predict vulnerability in C/C++ code using CodeBERT"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--code",
        type=str,
        help="Code string to analyze",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to source file to analyze",
    )
    parser.add_argument(
        "--input_json",
        type=str,
        help="Path to JSON file with list of code snippets",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save results (JSON)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for batch prediction",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device to use",
    )

    args = parser.parse_args()

    if not any([args.code, args.file, args.input_json]):
        parser.error("Must provide one of: --code, --file, or --input_json")

    print("Loading model...")
    predictor = VulnerabilityPredictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    print(f"Model loaded on {predictor.device}")

    if args.code:
        print("\nAnalyzing code snippet...")
        result = predictor.predict(args.code)
        print(f"\nResult:")
        print(f"  Prediction: {result['label']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Probabilities:")
        print(f"    Safe:       {result['probability']['safe']:.4f}")
        print(f"    Vulnerable: {result['probability']['vulnerable']:.4f}")

    elif args.file:
        print(f"\nAnalyzing file: {args.file}")
        result = predictor.predict_file(args.file)
        print(f"\nResult:")
        print(f"  Prediction: {result['label']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Probabilities:")
        print(f"    Safe:       {result['probability']['safe']:.4f}")
        print(f"    Vulnerable: {result['probability']['vulnerable']:.4f}")

    elif args.input_json:
        print(f"\nLoading codes from: {args.input_json}")
        with open(args.input_json, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            codes = data
        elif isinstance(data, dict) and "codes" in data:
            codes = data["codes"]
        else:
            raise ValueError("JSON must be a list or dict with 'codes' key")

        print(f"Analyzing {len(codes)} code snippets...")
        results = predictor.predict_batch(codes, batch_size=args.batch_size)

        vuln_count = sum(1 for r in results if r["prediction"] == 1)
        print(f"\nSummary:")
        print(f"  Total: {len(results)}")
        print(f"  Vulnerable: {vuln_count}")
        print(f"  Safe: {len(results) - vuln_count}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        else:
            result = results

    if args.output and (args.code or args.file):
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved to: {args.output}")


if __name__ == "__main__":
    main()
