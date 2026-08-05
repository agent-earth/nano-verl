import gc
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import cached_file

from .checkpoint import load_optimizer, save_optimizer
from .config import Config
from .grpo import compute_response_logprobs, grpo_loss


def _torch_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name in {"bf16", "bfloat16"} else torch.float16


class FSDPTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.outer_config = None
        self.processor_assets: dict[str, Path] = {}

    def load(self, model_path_or_name: str) -> None:
        config_path = Path(model_path_or_name) / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("model_type") == "qwen3_5":
                self.outer_config = config
        model_config = AutoConfig.from_pretrained(model_path_or_name, trust_remote_code=True)
        if self.outer_config is None and model_config.model_type == "qwen3_5":
            self.outer_config = model_config.to_dict()
        for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
            asset = config_path.parent / name
            if not asset.exists():
                resolved = cached_file(
                    model_path_or_name,
                    name,
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                )
                asset = Path(resolved) if resolved else asset
            if asset.exists():
                self.processor_assets[name] = asset
        dtype = _torch_dtype(self.cfg.model.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, trust_remote_code=True)
        model_config = getattr(model_config, "text_config", model_config)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path_or_name,
            trust_remote_code=True,
            dtype=dtype,
            config=model_config,
        ).cuda()
        self.model.config.use_cache = False
        if self.cfg.model.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        # The interface is FSDP-compatible; single GPU gains no sharding benefit.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.train.learning_rate,
            eps=1.0e-6,
            weight_decay=0.0,
        )
        load_optimizer(self.optimizer, model_path_or_name)

    def train_on_batch(self, batch: dict) -> dict[str, float]:
        self.model.train()
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in batch["prompts"]
        ]
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        active = [
            (prompt, response, advantage)
            for prompt, response, advantage in zip(prompts, batch["responses"], batch["advantages"])
            if abs(advantage) > self.cfg.algorithm.advantage_eps
        ]
        if not active:
            return {"loss": 0.0}
        sample_count = len(active)
        for prompt, response, advantage in active:
            logprob = compute_response_logprobs(
                self.model,
                self.tokenizer,
                [prompt],
                [response],
            )
            loss = grpo_loss(logprob, torch.tensor([advantage])) / sample_count
            loss.backward()
            total_loss += float(loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return {"loss": total_loss}

    def save(self, path: str | Path) -> str:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output, safe_serialization=True)
        self.tokenizer.save_pretrained(output)
        if self.outer_config is not None:
            (output / "config.json").write_text(
                json.dumps(self.outer_config, indent=2),
                encoding="utf-8",
            )
        for name, source in self.processor_assets.items():
            shutil.copy2(source, output / name)
        save_optimizer(self.optimizer, output)
        return str(output)

    def unload(self) -> None:
        # Trainer and vLLM take turns on one GPU, so every phase releases memory.
        del self.model, self.tokenizer, self.optimizer
        self.model = self.tokenizer = self.optimizer = None
        self.outer_config = None
        self.processor_assets = {}
        gc.collect()
        torch.cuda.empty_cache()
