import torch


def compute_group_advantages(
    grouped_rewards: list[list[float]], eps: float = 1.0e-6
) -> list[list[float]]:
    advantages = []
    for rewards in grouped_rewards:
        values = torch.tensor(rewards, dtype=torch.float32)
        # GRPO normalizes within each prompt group, removing the need for a critic.
        normalized = (values - values.mean()) / (values.std(unbiased=False) + eps)
        advantages.append(normalized.tolist())
    return advantages


def compute_response_logprobs(model, tokenizer, prompts: list[str], responses: list[str]) -> torch.Tensor:
    device = next(model.parameters()).device
    sequence_logprobs = []
    for prompt, response in zip(prompts, responses):
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        response_ids = tokenizer(response, add_special_tokens=False).input_ids
        if not response_ids:
            sequence_logprobs.append(torch.zeros((), device=device))
            continue

        input_ids = torch.tensor([prompt_ids + response_ids], device=device)
        logits = model(input_ids=input_ids).logits[:, :-1].float()
        targets = input_ids[:, 1:]
        token_logprobs = logits.log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        response_start = max(len(prompt_ids) - 1, 0)
        sequence_logprobs.append(token_logprobs[:, response_start:].mean())
    return torch.stack(sequence_logprobs)


def grpo_loss(logprobs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    return -(logprobs * advantages.to(logprobs.device)).mean()
