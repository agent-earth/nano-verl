from dataclasses import dataclass, fields
from pathlib import Path
from typing import TypeVar

import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3.5-0.8B"
    dtype: str = "fp16"
    gradient_checkpointing: bool = True


@dataclass
class TrainConfig:
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1.0e-6
    max_steps: int = 20
    save_every: int = 1
    eval_every: int = 5
    output_dir: str = "outputs/qwen_0_8b_grpo"
    seed: int = 42


@dataclass
class RolloutConfig:
    n: int = 4
    temperature: float = 1.0
    top_p: float = 0.9
    max_prompt_length: int = 128
    max_response_length: int = 32
    gpu_memory_utilization: float = 0.55


@dataclass
class AlgorithmConfig:
    kl_coef: float = 0.0
    advantage_eps: float = 1.0e-6


@dataclass
class DataConfig:
    source: str = "arithmetic"
    path: str | None = None
    shortest_first: bool = False
    train_size: int = 80
    eval_size: int = 30
    min_number: int = 0
    max_number: int = 20


@dataclass
class ReportConfig:
    enabled: bool = True
    max_cases: int = 8


@dataclass
class Config:
    model: ModelConfig
    train: TrainConfig
    rollout: RolloutConfig
    algorithm: AlgorithmConfig
    data: DataConfig
    report: ReportConfig


ConfigSection = TypeVar("ConfigSection")


def _load_section(section_type: type[ConfigSection], values: dict) -> ConfigSection:
    known = {field.name for field in fields(section_type)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown {section_type.__name__} fields: {sorted(unknown)}")
    return section_type(**values)


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = Config(
        model=_load_section(ModelConfig, raw.get("model", {})),
        train=_load_section(TrainConfig, raw.get("train", {})),
        rollout=_load_section(RolloutConfig, raw.get("rollout", {})),
        algorithm=_load_section(AlgorithmConfig, raw.get("algorithm", {})),
        data=_load_section(DataConfig, raw.get("data", {})),
        report=_load_section(ReportConfig, raw.get("report", {})),
    )
    if config.data.path and not Path(config.data.path).is_absolute():
        config.data.path = str((config_path.parent / config.data.path).resolve())
    return config
