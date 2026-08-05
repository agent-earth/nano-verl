from .config import Config
from .reward import compute_accuracy, compute_reward, extract_final_integer, is_correct
from .vllm_rollout import VLLMRollout


def evaluate(model_path: str, eval_dataset: list[dict[str, str]], cfg: Config) -> dict:
    rollout = VLLMRollout(model_path, cfg, n=1, deterministic=True)
    try:
        grouped = rollout.generate([item["prompt"] for item in eval_dataset])
    finally:
        rollout.unload()

    responses = [group[0] for group in grouped]
    answers = [item["answer"] for item in eval_dataset]
    rewards = [compute_reward(response, answer) for response, answer in zip(responses, answers)]
    examples = []
    for item, response, reward in zip(eval_dataset, responses, rewards):
        examples.append(
            {
                "prompt": item["prompt"],
                "answer": item["answer"],
                "response": response,
                "extracted_answer": extract_final_integer(response),
                "reward": reward,
                "correct": is_correct(response, item["answer"]),
            }
        )
    return {
        "accuracy": compute_accuracy(responses, answers),
        "mean_reward": sum(rewards) / len(rewards),
        "examples": examples,
    }
