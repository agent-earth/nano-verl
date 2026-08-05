#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from verl_mini.config import load_config
from verl_mini.data import build_eval_dataset, build_train_dataset, sample_prompts
from verl_mini.eval import evaluate
from verl_mini.fsdp_trainer import FSDPTrainer
from verl_mini.grpo import compute_group_advantages
from verl_mini.report import generate_report
from verl_mini.reward import compute_accuracy, compute_reward
from verl_mini.utils import append_jsonl, resolve_model_path, set_seed, write_json
from verl_mini.vllm_rollout import VLLMRollout


def save_initial_checkpoint(model_path: str, checkpoint_path: Path, cfg) -> str:
    trainer = FSDPTrainer(cfg)
    trainer.load(model_path)
    try:
        return trainer.save(checkpoint_path)
    finally:
        trainer.unload()


def flatten_rollouts(samples: list[dict[str, str]], grouped_responses: list[list[str]], cfg) -> dict:
    grouped_rewards = []
    prompts = []
    responses = []
    answers = []
    for sample, response_group in zip(samples, grouped_responses):
        rewards = [compute_reward(response, sample["answer"]) for response in response_group]
        grouped_rewards.append(rewards)
        prompts.extend([sample["prompt"]] * len(response_group))
        responses.extend(response_group)
        answers.extend([sample["answer"]] * len(response_group))
    grouped_advantages = compute_group_advantages(grouped_rewards, cfg.algorithm.advantage_eps)
    return {
        "prompts": prompts,
        "responses": responses,
        "answers": answers,
        "rewards": [reward for group in grouped_rewards for reward in group],
        "advantages": [advantage for group in grouped_advantages for advantage in group],
    }


def replace_checkpoint(source: Path, destination: Path) -> None:
    # Checkpoint sync is deliberately file-based: slow, explicit, and easy to inspect.
    if destination.exists():
        shutil.rmtree(destination)
    source.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train verl-mini GRPO on one GPU.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    output_dir = Path(cfg.train.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    eval_before_path = output_dir / "eval_before.json"
    eval_after_path = output_dir / "eval_after.json"
    latest_checkpoint = output_dir / "latest"
    best_checkpoint = output_dir / "best"
    metrics_path.unlink(missing_ok=True)

    train_dataset = build_train_dataset(cfg)
    eval_dataset = build_eval_dataset(cfg)
    model_path = resolve_model_path(cfg.model.name)
    print(f"model_path={model_path}", flush=True)
    print("initializing checkpoint", flush=True)
    save_initial_checkpoint(model_path, latest_checkpoint, cfg)

    # Baseline and final eval use the exact same deterministic eval examples.
    baseline = evaluate(str(latest_checkpoint), eval_dataset, cfg)
    write_json(eval_before_path, baseline)
    print(f"baseline_eval_accuracy={baseline['accuracy']:.6f}", flush=True)
    print(f"baseline_eval_reward={baseline['mean_reward']:.6f}", flush=True)

    final_eval = baseline
    best_eval = baseline
    best_step = 0
    for step in range(1, cfg.train.max_steps + 1):
        prompt_batch_size = (
            cfg.train.micro_batch_size * cfg.train.gradient_accumulation_steps
        )
        samples = sample_prompts(train_dataset, prompt_batch_size, step - 1)
        rollout = VLLMRollout(str(latest_checkpoint), cfg)
        try:
            grouped_responses = rollout.generate([sample["prompt"] for sample in samples])
        finally:
            rollout.unload()
        batch = flatten_rollouts(samples, grouped_responses, cfg)

        trainer = FSDPTrainer(cfg)
        trainer.load(str(latest_checkpoint))
        try:
            train_metrics = trainer.train_on_batch(batch)
            next_checkpoint = output_dir / "next"
            trainer.save(next_checkpoint)
        finally:
            trainer.unload()
        replace_checkpoint(next_checkpoint, latest_checkpoint)

        train_mean_reward = sum(batch["rewards"]) / len(batch["rewards"])
        train_accuracy = compute_accuracy(batch["responses"], batch["answers"])
        eval_result = None
        if step % cfg.train.eval_every == 0 or step == cfg.train.max_steps:
            eval_result = evaluate(str(latest_checkpoint), eval_dataset, cfg)
            final_eval = eval_result
            print(f"eval_reward={eval_result['mean_reward']:.6f}", flush=True)
            print(f"eval_accuracy={eval_result['accuracy']:.6f}", flush=True)
            if eval_result["accuracy"] > best_eval["accuracy"]:
                if best_checkpoint.exists():
                    shutil.rmtree(best_checkpoint)
                shutil.copytree(latest_checkpoint, best_checkpoint)
                best_eval = eval_result
                best_step = step

        metric = {
            "step": step,
            "train_mean_reward": train_mean_reward,
            "train_accuracy": train_accuracy,
            "loss": train_metrics["loss"],
            "eval_mean_reward": None if eval_result is None else eval_result["mean_reward"],
            "eval_accuracy": None if eval_result is None else eval_result["accuracy"],
            "checkpoint_path": str(latest_checkpoint),
        }
        append_jsonl(metrics_path, metric)
        print(
            f"step={step} current_train_reward={train_mean_reward:.6f} "
            f"train_accuracy={train_accuracy:.6f} loss={train_metrics['loss']:.6f} "
            f"checkpoint={latest_checkpoint}",
            flush=True,
        )

    write_json(eval_after_path, final_eval)
    report_path = ""
    if cfg.report.enabled:
        report_path = generate_report(
            cfg,
            metrics_path,
            eval_before_path,
            eval_after_path,
            output_dir,
        )

    accuracy_improvement = final_eval["accuracy"] - baseline["accuracy"]
    reward_improvement = final_eval["mean_reward"] - baseline["mean_reward"]
    print(f"final_eval_accuracy={final_eval['accuracy']:.6f}", flush=True)
    print(f"accuracy_improvement={accuracy_improvement:.6f}", flush=True)
    print(f"final_eval_reward={final_eval['mean_reward']:.6f}", flush=True)
    print(f"reward_improvement={reward_improvement:.6f}", flush=True)
    print(f"improvement={reward_improvement:.6f}", flush=True)
    print(f"best_eval_step={best_step}", flush=True)
    print(f"best_eval_accuracy={best_eval['accuracy']:.6f}", flush=True)
    print(f"best_eval_reward={best_eval['mean_reward']:.6f}", flush=True)
    print(f"best_checkpoint={best_checkpoint if best_step else ''}", flush=True)
    print(f"report_path={report_path}", flush=True)


if __name__ == "__main__":
    main()
