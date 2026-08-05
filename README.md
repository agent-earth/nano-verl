# nano-verl

`nano-verl` is a single-GPU MVP that shows engine-level training/inference
disaggregation with the smallest practical amount of code. It is designed for
teaching and readability, not throughput or feature parity with verl.

The default demo uses:

- Qwen/Qwen3.5-0.8B, preferring a nearby local `Qwen3.5-0.8B` directory.
- A PyTorch `FSDPTrainer`-compatible interface on one GPU.
- vLLM for rollout and deterministic evaluation.
- A minimal GRPO policy-gradient loss without PPO, a critic, or a reference model.
- Deterministic three-digit addition/subtraction with an exact-match rule reward.

## How it works

Trainer and vLLM do not stay resident together. Each step is intentionally
sequential:

1. vLLM loads the latest checkpoint and generates grouped responses.
2. vLLM unloads and releases CUDA memory.
3. PyTorch loads the checkpoint, applies one GRPO update, and saves it.
4. PyTorch unloads before the next rollout or evaluation.

Weights are synchronized through checkpoints. This is slower than hot weight
updates, but it is explicit, stable, and easy to inspect in a teaching project.
The class is named `FSDPTrainer` to keep a future multi-GPU extension point;
single-GPU execution uses a normal PyTorch model because sharding has no benefit.

GRPO is used instead of PPO because it normalizes rewards within each prompt
group and does not require a critic/value model. This implementation is a
teaching mini-GRPO: it uses `-mean(response_logprob * advantage)` and does not
implement PPO ratios or an industrial reference-policy KL path.

The rule reward avoids a separate reward model. Correct final integers receive
`1.0`; parseable but incorrect integers receive a small `0.1` format reward.
Evaluation accuracy always uses strict exact match.

## Run

Python 3.10+, one CUDA GPU, and the packages in `requirements.txt` are required.
From this project directory:

```bash
python examples/train_grpo_single_gpu.py \
  --config configs/qwen_0_8b_single_gpu.yaml
```

The script automatically sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`. To use a
different local model checkpoint:

```bash
VERL_MINI_MODEL=/path/to/Qwen3.5-0.8B \
python examples/train_grpo_single_gpu.py \
  --config configs/qwen_0_8b_single_gpu.yaml
```

The default config is intentionally short and was verified on one NVIDIA A10:
three steps, four prompts per step, four responses per prompt, and a fixed
30-case evaluation set.

Two additional reproducible configs are included:

```bash
# Ten-step arithmetic run with eval every two steps.
python examples/train_grpo_single_gpu.py \
  --config configs/qwen_0_8b_arithmetic_long.yaml

# Ten-step local GSM8K run. The path is resolved relative to the config file.
python examples/train_grpo_single_gpu.py \
  --config configs/qwen_0_8b_gsm8k.yaml
```

The GSM8K config deterministically selects the shortest 80 train questions and
shortest 20 test questions. Selection depends only on question length, never on
model outputs. It is a small stability test, not a full GSM8K benchmark.
GSM8K has much higher variance than the synthetic task: use the fixed test
accuracy and before/after cases, and keep the best eval checkpoint rather than
assuming more steps are always better.

Verified A10 results from the included configs:

| Run | Steps | Fixed eval accuracy | Mean reward |
| --- | ---: | ---: | ---: |
| Arithmetic long | 10 | 0.367 -> 0.800 | 0.430 -> 0.820 |
| GSM8K shortest fixed subset | 10 | 0.650 -> 0.700 | 0.685 -> 0.725 |

The GSM8K result is only 20 deterministic test cases and one case changed from
wrong to correct, so treat it as a pipeline demonstration rather than a claim
about full-dataset GSM8K performance.

## Verify learning

The command prints:

- `baseline_eval_accuracy` and `baseline_eval_reward` before training.
- `current_train_reward`, train accuracy, and loss for every step.
- `eval_reward` and eval accuracy at configured evaluation steps.
- final accuracy/reward, both improvements, and `report_path`.

Training is effective when final exact-match accuracy and reward exceed their
baseline values. Do not infer success from loss alone. Open
`outputs/qwen_0_8b_grpo/report.html` directly in a browser and inspect:

- reward, loss, and accuracy curves;
- the final summary;
- fixed evaluation cases before and after training.

The report and metrics always contain measured values. If a run does not
improve, the script reports that result rather than fabricating a gain.

Each run generates:

```text
outputs/qwen_0_8b_grpo/
  metrics.jsonl
  eval_before.json
  eval_after.json
  report.html
  figures/
    reward_curve.png
    loss_curve.png
    accuracy_curve.png
  latest/
  best/                 # only created after an eval accuracy improvement
```

## Tuning

If GPU memory is insufficient:

- reduce `rollout.max_response_length` to `2`;
- reduce `rollout.n` to `2`;
- reduce `rollout.max_prompt_length` to `64`;
- reduce `train.micro_batch_size` or `train.max_steps`;
- use a smaller model.

If learning is weak or noisy:

- increase `train.max_steps`;
- increase `rollout.n`;
- try learning rates `2e-6` or `5e-6`;
- simplify the range, for example `data.max_number: 100`;
- keep responses short so rewards target the final integer;
- inspect extracted answers in `eval_before.json` and `eval_after.json`.

## Layout

```text
verl_mini/
  config.py        YAML dataclasses
  data.py          deterministic train/eval arithmetic data
  reward.py        integer parser, rule reward, exact-match accuracy
  grpo.py          group advantages, response log-probs, GRPO loss
  fsdp_trainer.py  single-GPU FSDP-compatible PyTorch trainer
  vllm_rollout.py  load/generate/unload vLLM phase
  checkpoint.py    optimizer checkpoint helpers
  eval.py          deterministic fixed-set evaluation
  report.py        PNG curves and standalone HTML report
  utils.py         seed, model resolution, JSON helpers
examples/train_grpo_single_gpu.py
configs/qwen_0_8b_single_gpu.yaml
```

Future extensions can place an FSDP trainer on GPU 0 and vLLM on GPU 1, or use
GPU 0-1 for FSDP, GPU 2 for rollout, and GPU 3 for evaluation/reward.
