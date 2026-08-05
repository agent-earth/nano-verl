import re
from decimal import Decimal, InvalidOperation


NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def extract_final_integer(text: str) -> str | None:
    answer_text = text.rsplit("####", 1)[-1] if "####" in text else text
    matches = NUMBER_PATTERN.findall(answer_text)
    if not matches:
        return None
    try:
        return str(Decimal(matches[-1].replace(",", "")).normalize())
    except InvalidOperation:
        return None


def normalize_answer(answer: str) -> str | None:
    return extract_final_integer(answer)


def is_correct(response: str, answer: str) -> bool:
    return extract_final_integer(response) == normalize_answer(answer)


def compute_reward(response: str, answer: str) -> float:
    # A rule reward avoids a reward model, keeping this single-GPU demo small.
    extracted = extract_final_integer(response)
    if extracted == normalize_answer(answer):
        return 1.0
    return 0.1 if extracted is not None else 0.0


def compute_accuracy(responses: list[str], answers: list[str]) -> float:
    if not responses:
        return 0.0
    correct = sum(is_correct(response, answer) for response, answer in zip(responses, answers))
    return correct / len(responses)
