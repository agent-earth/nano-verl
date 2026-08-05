import html
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_metrics(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _plot(path: Path, title: str, ylabel: str, series: list[tuple[str, list, list]]) -> None:
    plt.figure(figsize=(7, 4))
    for label, steps, values in series:
        points = [(step, value) for step, value in zip(steps, values) if value is not None]
        if points:
            plt.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                label=label,
            )
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def _case_rows(before: dict, after: dict, max_cases: int) -> str:
    pairs = list(zip(before["examples"], after["examples"]))
    categories = [(False, True), (False, False), (True, True), (True, False)]
    selected = []
    for category in categories:
        match = next(
            (
                pair
                for pair in pairs
                if (pair[0]["correct"], pair[1]["correct"]) == category
            ),
            None,
        )
        if match is not None:
            selected.append(match)
    for pair in pairs:
        if pair not in selected:
            selected.append(pair)
    selected = selected[:max_cases]
    rows = []
    for old, new in selected:
        change = f"{'correct' if old['correct'] else 'wrong'} -> {'correct' if new['correct'] else 'wrong'}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(old['prompt'])}</td>"
            f"<td>{html.escape(old['answer'])}</td>"
            f"<td><pre>{html.escape(old['response'])}</pre><small>{old['extracted_answer']}</small></td>"
            f"<td>{old['correct']}</td>"
            f"<td><pre>{html.escape(new['response'])}</pre><small>{new['extracted_answer']}</small></td>"
            f"<td>{new['correct']}</td>"
            f"<td>{change}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _metric_cell(value: float | None, digits: int = 3) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def generate_report(
    config: Config,
    metrics_path: str | Path,
    eval_before_path: str | Path,
    eval_after_path: str | Path,
    output_dir: str | Path,
) -> str:
    output = Path(output_dir)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    metrics = _read_metrics(metrics_path)
    before = _read_json(eval_before_path)
    after = _read_json(eval_after_path)
    steps = [item["step"] for item in metrics]

    _plot(
        figures / "reward_curve.png",
        "Reward",
        "mean reward",
        [
            ("train", steps, [item["train_mean_reward"] for item in metrics]),
            (
                "eval",
                [0] + steps,
                [before["mean_reward"]] + [item["eval_mean_reward"] for item in metrics],
            ),
        ],
    )
    _plot(
        figures / "loss_curve.png",
        "GRPO Loss",
        "loss",
        [("loss", steps, [item["loss"] for item in metrics])],
    )
    _plot(
        figures / "accuracy_curve.png",
        "Exact-Match Accuracy",
        "accuracy",
        [
            ("train", steps, [item["train_accuracy"] for item in metrics]),
            (
                "eval",
                [0] + steps,
                [before["accuracy"]] + [item["eval_accuracy"] for item in metrics],
            ),
        ],
    )

    final_loss = metrics[-1]["loss"] if metrics else None
    first_loss = metrics[0]["loss"]
    accuracy_improvement = after["accuracy"] - before["accuracy"]
    reward_improvement = after["mean_reward"] - before["mean_reward"]
    loss_direction = "decreased" if final_loss < first_loss else "increased"
    if any(
        (metrics[index]["loss"] - metrics[index - 1]["loss"])
        * (metrics[index + 1]["loss"] - metrics[index]["loss"])
        < 0
        for index in range(1, len(metrics) - 1)
    ):
        loss_direction = "fluctuated"
    analysis = (
        f"Accuracy changed by {accuracy_improvement:+.3f} and mean reward changed by "
        f"{reward_improvement:+.3f}. GRPO loss {loss_direction} and ranged from "
        f"{min(item['loss'] for item in metrics):.4f} to "
        f"{max(item['loss'] for item in metrics):.4f}, with final loss {final_loss:.4f}. "
    )
    if accuracy_improvement > 0:
        analysis += "The fixed eval set improved after training."
    elif accuracy_improvement == 0:
        analysis += (
            "Exact-match accuracy was unchanged. Any reward change only reflects "
            "format/parseability and is not evidence of better task accuracy."
        )
    else:
        analysis += (
            "Exact-match accuracy decreased. Try more steps, a larger rollout group, "
            "a higher learning rate, or a smaller arithmetic range."
        )

    config_items = asdict(config)
    if config_items["data"].get("path"):
        config_items["data"]["path"] = "<local dataset>"
    baseline_row = (
        "<tr><td>baseline</td><td></td><td></td><td></td>"
        f"<td>{before['mean_reward']:.3f}</td><td>{before['accuracy']:.3f}</td></tr>"
    )
    metric_rows = "\n".join(
        "<tr>"
        f"<td>{item['step']}</td><td>{item['train_mean_reward']:.3f}</td>"
        f"<td>{item['train_accuracy']:.3f}</td><td>{item['loss']:.4f}</td>"
        f"<td>{_metric_cell(item['eval_mean_reward'])}</td>"
        f"<td>{_metric_cell(item['eval_accuracy'])}</td>"
        "</tr>"
        for item in metrics
        if item["step"] == metrics[-1]["step"] or item["eval_accuracy"] is not None
    )
    table_rows = baseline_row + metric_rows
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>verl-mini report</title>
<style>
body{{font:15px/1.5 sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#18211d}}
h1,h2{{color:#075f4e}} table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}
th,td{{border:1px solid #ccd6d1;padding:8px;text-align:left;vertical-align:top}}
pre{{white-space:pre-wrap;max-width:360px;margin:0}} img{{max-width:100%;border:1px solid #ddd}}
.summary{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
.summary div{{border:1px solid #ccd6d1;padding:12px}} code{{white-space:pre-wrap}}
</style></head><body>
<h1>verl-mini Training Report</h1>
<h2>Run Configuration</h2>
<p><strong>Project:</strong> verl-mini</p>
<pre><code>{html.escape(json.dumps(config_items, indent=2))}</code></pre>
<h2>Final Summary</h2>
<div class="summary">
<div>Baseline accuracy<br><strong>{before['accuracy']:.3f}</strong></div>
<div>Final accuracy<br><strong>{after['accuracy']:.3f}</strong></div>
<div>Accuracy improvement<br><strong>{accuracy_improvement:+.3f}</strong></div>
<div>Baseline reward<br><strong>{before['mean_reward']:.3f}</strong></div>
<div>Final reward<br><strong>{after['mean_reward']:.3f}</strong></div>
<div>Reward improvement<br><strong>{reward_improvement:+.3f}</strong></div>
<div>Final loss<br><strong>{final_loss:.4f}</strong></div>
</div>
<h2>Training Curves</h2>
<img src="figures/reward_curve.png" alt="reward curve">
<img src="figures/loss_curve.png" alt="loss curve">
<img src="figures/accuracy_curve.png" alt="accuracy curve">
<h2>Training Metrics</h2>
<table><thead><tr><th>step</th><th>train reward</th><th>train accuracy</th>
<th>loss</th><th>eval reward</th><th>eval accuracy</th></tr></thead><tbody>{table_rows}</tbody></table>
<h2>Before / After Cases</h2>
<table><thead><tr><th>prompt</th><th>gold</th><th>before</th><th>before correct</th>
<th>after</th><th>after correct</th><th>change</th>
</tr></thead><tbody>{_case_rows(before, after, config.report.max_cases)}</tbody></table>
<h2>Analysis</h2><p>{html.escape(analysis)}</p>
</body></html>"""
    report_path = output / "report.html"
    report_path.write_text(report, encoding="utf-8")
    return str(report_path)
