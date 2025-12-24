import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "microsoft/codebert-base"
    head_type: str = "mlp"
    num_labels: int = 2
    dropout: float = 0.1
    freeze_encoder: bool = False


@dataclass
class DataConfig:
    max_length: int = 512
    code_column: str = "func_clean"
    label_column: str = "target"
    cache_dir: str = "./cache/codebert"


@dataclass
class TrainingConfig:
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_epochs: int = 10
    early_stopping_patience: int = 3
    fp16: bool = True
    seed: int = 42


@dataclass
class PathsConfig:
    train_data: str = "/kaggle/input/devign-dataset/train.parquet"
    valid_data: str = "/kaggle/input/devign-dataset/valid.parquet"
    test_data: str = "/kaggle/input/devign-dataset/test.parquet"
    output_dir: str = "./output/codebert"
    checkpoint_dir: str = "./checkpoints/codebert"


@dataclass
class CodeBERTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "CodeBERTConfig":
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**config_dict.get("data", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            paths=PathsConfig(**config_dict.get("paths", {})),
        )

    def to_dict(self) -> dict:
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "paths": self.paths.__dict__,
        }


def load_config(
    config_path: Optional[str] = None,
    args: Optional[argparse.Namespace] = None,
) -> CodeBERTConfig:
    config_dict = {}

    if config_path is None:
        config_path = Path(__file__).parent / "default.yaml"

    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

    config = CodeBERTConfig.from_dict(config_dict)

    if args is not None:
        _override_from_args(config, args)

    return config


def _override_from_args(config: CodeBERTConfig, args: argparse.Namespace) -> None:
    arg_dict = vars(args)

    if arg_dict.get("model_name"):
        config.model.name = arg_dict["model_name"]
    if arg_dict.get("head_type"):
        config.model.head_type = arg_dict["head_type"]
    if arg_dict.get("dropout") is not None:
        config.model.dropout = arg_dict["dropout"]
    if arg_dict.get("freeze_encoder") is not None:
        config.model.freeze_encoder = arg_dict["freeze_encoder"]

    if arg_dict.get("max_length"):
        config.data.max_length = arg_dict["max_length"]
    if arg_dict.get("code_column"):
        config.data.code_column = arg_dict["code_column"]

    if arg_dict.get("batch_size"):
        config.training.batch_size = arg_dict["batch_size"]
    if arg_dict.get("learning_rate"):
        config.training.learning_rate = arg_dict["learning_rate"]
    if arg_dict.get("max_epochs"):
        config.training.max_epochs = arg_dict["max_epochs"]
    if arg_dict.get("seed"):
        config.training.seed = arg_dict["seed"]
    if arg_dict.get("fp16") is not None:
        config.training.fp16 = arg_dict["fp16"]

    if arg_dict.get("train_data"):
        config.paths.train_data = arg_dict["train_data"]
    if arg_dict.get("valid_data"):
        config.paths.valid_data = arg_dict["valid_data"]
    if arg_dict.get("test_data"):
        config.paths.test_data = arg_dict["test_data"]
    if arg_dict.get("output_dir"):
        config.paths.output_dir = arg_dict["output_dir"]
    if arg_dict.get("checkpoint_dir"):
        config.paths.checkpoint_dir = arg_dict["checkpoint_dir"]


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CodeBERT Vulnerability Detection")

    parser.add_argument("--config", type=str, help="Path to config YAML file")

    parser.add_argument("--model_name", type=str, help="Model name or path")
    parser.add_argument("--head_type", type=str, choices=["mlp", "cnn"])
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--freeze_encoder", action="store_true")

    parser.add_argument("--max_length", type=int)
    parser.add_argument("--code_column", type=str)

    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_fp16", dest="fp16", action="store_false")

    parser.add_argument("--train_data", type=str)
    parser.add_argument("--valid_data", type=str)
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--checkpoint_dir", type=str)

    return parser
