import random
import re
from pathlib import Path

import pandas as pd

from .config import Config


Example = dict[str, str]
GSM8K_ANSWER = re.compile(r"####\s*([^\n]+)")


def _build_dataset(cfg: Config, size: int, seed: int, excluded: set[str]) -> list[Example]:
    rng = random.Random(seed)
    dataset: list[Example] = []
    seen = set(excluded)
    while len(dataset) < size:
        left = rng.randint(cfg.data.min_number, cfg.data.max_number)
        right = rng.randint(cfg.data.min_number, cfg.data.max_number)
        operator = rng.choice(["+", "-"])
        answer = left + right if operator == "+" else left - right
        prompt = f"Compute {left} {operator} {right}. Return only the final integer answer."
        if prompt in seen:
            continue
        seen.add(prompt)
        dataset.append({"prompt": prompt, "answer": str(answer)})
    return dataset


def _gsm8k_dataset(cfg: Config, split: str, size: int) -> list[Example]:
    if not cfg.data.path:
        raise ValueError("data.path is required when data.source is gsm8k")
    path = Path(cfg.data.path).expanduser() / "main" / f"{split}-00000-of-00001.parquet"
    frame = pd.read_parquet(path)
    if cfg.data.shortest_first:
        frame = frame.assign(question_length=frame.question.str.len()).sort_values(
            ["question_length", "question"],
            kind="stable",
        )
    else:
        frame = frame.sample(frac=1.0, random_state=cfg.train.seed)
    frame = frame.head(size)
    dataset = []
    for row in frame.itertuples(index=False):
        match = GSM8K_ANSWER.search(row.answer)
        if not match:
            continue
        prompt = (
            f"{row.question}\n"
            "Solve in at most three short lines. The last line must be "
            "'####' followed by only the final numeric answer."
        )
        dataset.append({"prompt": prompt, "answer": match.group(1).strip()})
    if len(dataset) != size:
        raise ValueError(f"Expected {size} GSM8K rows, found {len(dataset)}")
    return dataset


def build_train_dataset(cfg: Config) -> list[Example]:
    if cfg.data.source == "gsm8k":
        return _gsm8k_dataset(cfg, "train", cfg.data.train_size)
    if cfg.data.source != "arithmetic":
        raise ValueError(f"Unknown data source: {cfg.data.source}")
    return _build_dataset(cfg, cfg.data.train_size, cfg.train.seed, set())


def build_eval_dataset(cfg: Config) -> list[Example]:
    if cfg.data.source == "gsm8k":
        return _gsm8k_dataset(cfg, "test", cfg.data.eval_size)
    train_prompts = {item["prompt"] for item in build_train_dataset(cfg)}
    return _build_dataset(cfg, cfg.data.eval_size, cfg.train.seed + 1, train_prompts)


def sample_prompts(dataset: list[Example], batch_size: int, step: int) -> list[Example]:
    start = (step * batch_size) % len(dataset)
    return [dataset[(start + index) % len(dataset)] for index in range(batch_size)]
