import gc

import torch

from .config import Config


def _vllm_dtype(name: str) -> str:
    return {"fp16": "float16", "bf16": "bfloat16"}.get(name, name)


class VLLMRollout:
    def __init__(self, model_path: str, cfg: Config, n: int | None = None, deterministic: bool = False):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError("vLLM is required. Install it with `pip install vllm`.") from exc

        self.SamplingParams = SamplingParams
        self.cfg = cfg
        self.n = n if n is not None else cfg.rollout.n
        self.deterministic = deterministic
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype=_vllm_dtype(cfg.model.dtype),
            tensor_parallel_size=1,
            gpu_memory_utilization=cfg.rollout.gpu_memory_utilization,
            max_model_len=cfg.rollout.max_prompt_length + cfg.rollout.max_response_length,
            hf_overrides={"architectures": ["Qwen3_5ForCausalLM"]},
            language_model_only=True,
            enforce_eager=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, prompts: list[str]) -> list[list[str]]:
        rendered = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        sampling = self.SamplingParams(
            n=self.n,
            temperature=0.0 if self.deterministic else self.cfg.rollout.temperature,
            top_p=1.0 if self.deterministic else self.cfg.rollout.top_p,
            max_tokens=self.cfg.rollout.max_response_length,
            seed=self.cfg.train.seed,
        )
        outputs = self.llm.generate(rendered, sampling, use_tqdm=False)
        return [[candidate.text for candidate in output.outputs] for output in outputs]

    def unload(self) -> None:
        # vLLM cannot remain resident while the trainer uses the same single GPU.
        engine = getattr(self.llm, "llm_engine", None)
        if engine is not None and hasattr(engine, "shutdown"):
            engine.shutdown()
        del self.llm
        self.llm = None
        gc.collect()
        torch.cuda.empty_cache()
