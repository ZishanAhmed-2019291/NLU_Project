import json
import math
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm.auto import tqdm

from model import GPT2Scratch, init_weights, count_parameters
from utils import make_dataloaders


@dataclass
class ExperimentConfig:
    name: str
    lr: float
    d_model: int = 128
    n_heads: int = 2
    num_layers: int = 2
    ff_dim: int = 512
    dropout: float = 0.0
    tie_weights: bool = False
    block_size: int = 128
    batch_size_train: int = 32
    batch_size_eval: int = 128
    epochs: int = 10
    patience: int = 4
    weight_decay: float = 0.01
    grad_clip: float = 1.0


def train_one_epoch(model, loader, optimizer, pad_token_id: int, grad_clip: float = 1.0, max_batches=None):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for step, (input_ids, labels, n_tokens) in enumerate(pbar, start=1):
        if max_batches is not None and step > max_batches:
            break

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=pad_token_id,
            reduction="sum",
        )
        token_count = int(n_tokens.item())
        loss = loss_sum / max(token_count, 1)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += float(loss_sum.item())
        total_tokens += token_count
        pbar.set_postfix(loss=total_loss / max(total_tokens, 1))

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_ppl(model, loader, pad_token_id: int, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for step, (input_ids, labels, n_tokens) in enumerate(pbar, start=1):
        if max_batches is not None and step > max_batches:
            break

        logits = model(input_ids)
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=pad_token_id,
            reduction="sum",
        )

        total_loss += float(loss_sum.item())
        total_tokens += int(n_tokens.item())

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    return ppl, avg_loss


def build_model(config: ExperimentConfig, vocab_size: int, device: str):
    model = GPT2Scratch(
        vocab_size=vocab_size,
        pos_emb_size=config.block_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        num_layers=config.num_layers,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
        tie_weights=config.tie_weights,
    ).to(device)
    model.apply(init_weights)
    return model


def fit_one_experiment(config: ExperimentConfig, tokenizer, splits, paths, device: str, quick: bool = False):
    print("\n" + "=" * 90)
    print("Experiment:", config.name)
    print(config)

    train_raw, dev_raw, test_raw = splits
    train_loader, dev_loader, test_loader = make_dataloaders(
        train_raw=train_raw,
        dev_raw=dev_raw,
        test_raw=test_raw,
        tokenizer=tokenizer,
        device=device,
        batch_size_train=config.batch_size_train,
        batch_size_eval=config.batch_size_eval,
        block_size=config.block_size,
        quick=quick,
    )

    model = build_model(config, vocab_size=len(tokenizer), device=device)
    total_params, trainable_params = count_parameters(model)
    print(f"Parameters: total={total_params:,}, trainable={trainable_params:,}")
    print("Weight tying active:", model.lm_head.weight is model.token_embed.weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_dev_ppl = float("inf")
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    history = []

    max_train_batches = 40 if quick else None
    max_eval_batches = 20 if quick else None
    n_epochs = min(config.epochs, 2) if quick else config.epochs

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            tokenizer.pad_token_id,
            grad_clip=config.grad_clip,
            max_batches=max_train_batches,
        )
        dev_ppl, dev_loss = evaluate_ppl(
            model,
            dev_loader,
            tokenizer.pad_token_id,
            max_batches=max_eval_batches,
        )

        history.append({
            "name": config.name,
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_ppl": dev_ppl,
        })
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | dev_loss={dev_loss:.4f} | dev_ppl={dev_ppl:.2f}")

        if dev_ppl < best_dev_ppl:
            best_dev_ppl = dev_ppl
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_ppl, test_loss = evaluate_ppl(
        model,
        test_loader,
        tokenizer.pad_token_id,
        max_batches=max_eval_batches,
    )

    ckpt_path = paths["tmp_ckpt_dir"] / f"{config.name}.pt"
    torch.save(
        {
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": asdict(config),
            "tokenizer_name": "openai-community/gpt2",
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "best_dev_ppl": best_dev_ppl,
            "test_ppl": test_ppl,
            "best_epoch": best_epoch,
        },
        ckpt_path,
    )

    print(f"Best dev PPL: {best_dev_ppl:.2f}")
    print(f"Test PPL: {test_ppl:.2f}")

    result = asdict(config)
    result.update({
        "total_params": total_params,
        "trainable_params": trainable_params,
        "best_epoch": best_epoch,
        "best_dev_ppl": best_dev_ppl,
        "test_loss": test_loss,
        "test_ppl": test_ppl,
        "checkpoint_path": str(ckpt_path),
        "quick": quick,
    })
    return result, pd.DataFrame(history)


def run_experiments(experiments, tokenizer, splits, paths, device: str, quick: bool = False):
    results = []
    histories = {}

    for config in experiments:
        result, history = fit_one_experiment(config, tokenizer, splits, paths, device, quick=quick)
        results.append(result)
        histories[config.name] = history

    results_df = pd.DataFrame(results)
    history_df = pd.concat(histories.values(), ignore_index=True) if histories else pd.DataFrame()
    return results_df, history_df


def save_best_model(results_df, paths):
    best_row = results_df.sort_values("best_dev_ppl").iloc[0]

    for existing_file in paths["bin_dir"].glob("*"):
        if existing_file.is_file():
            existing_file.unlink()
        elif existing_file.is_dir():
            shutil.rmtree(existing_file)

    best_checkpoint_path = Path(best_row["checkpoint_path"])
    final_model_path = paths["bin_dir"] / "part1a_best_model.pt"
    shutil.copy2(best_checkpoint_path, final_model_path)

    metadata = best_row.to_dict()
    metadata.update({
        "selection_metric": "best_dev_ppl",
        "final_model_path": str(final_model_path),
    })
    metadata_path = paths["results_dir"] / "part1a_best_model_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Best Part 1.A configuration selected by development PPL:")
    print(best_row[[
        "name", "lr", "d_model", "n_heads", "num_layers", "ff_dim", "dropout", "tie_weights",
        "best_epoch", "best_dev_ppl", "test_ppl", "total_params"
    ]])
    print("\nSaved best model:", final_model_path)
    print("Saved metadata:", metadata_path)
    return best_row, final_model_path, metadata_path


PAPER = {
    "blue": "#1F4E79",
    "sky": "#4C78A8",
    "orange": "#F28E2B",
    "green": "#59A14F",
    "red": "#E15759",
    "purple": "#6B4C9A",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "dark": "#111827",
}


def configure_matplotlib():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.8,
    })


def short_name(name):
    mapping = {
        "A0_baseline_chosen_lr": "A0 baseline",
        "A1_change_d_model_192": "A1 d=192",
        "A2_change_n_heads_4": "A2 heads=4",
        "A3_change_num_layers_3": "A3 layers=3",
        "A4_change_ff_dim_768": "A4 FF=768",
        "A5_add_dropout_0_1": "A5 dropout",
        "A6_add_weight_tying": "A6 weight tying",
    }
    if name in mapping:
        return mapping[name]
    return (
        name.replace("A0_", "A0 ")
        .replace("A1_", "A1 ")
        .replace("A2_", "A2 ")
        .replace("A3_", "A3 ")
        .replace("A4_", "A4 ")
        .replace("A5_", "A5 ")
        .replace("A6_", "A6 ")
        .replace("_", " ")
    )


def save_current_figure(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    print("Saved:", path)
    plt.close()


def prepare_history(history_df):
    df = history_df.copy()
    if "train_ppl" not in df.columns:
        df["train_ppl"] = np.exp(np.minimum(df["train_loss"], 20))
    df["generalization_gap"] = df["dev_ppl"] - df["train_ppl"]
    df["label"] = df["name"].map(short_name)
    return df


def plot_lr_search_absolute(lr_df, figures_dir):
    plot_df = lr_df.sort_values("lr").copy()
    plot_df["lr_label"] = plot_df["lr"].map(lambda x: f"{x:.0e}")
    best_idx = plot_df["best_dev_ppl"].idxmin()
    colors = [PAPER["orange"] if i == best_idx else PAPER["sky"] for i in plot_df.index]

    x = np.arange(len(plot_df))
    width = 0.36
    _, ax = plt.subplots(figsize=(9.8, 5.3))
    ax.bar(x - width / 2, plot_df["best_dev_ppl"], width, color=colors, label="Best dev PPL")
    ax.bar(x + width / 2, plot_df["test_ppl"], width, color=PAPER["green"], alpha=0.88, label="Test PPL")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["lr_label"])
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Perplexity")
    ax.set_title("Baseline learning-rate search")
    ax.legend(frameon=False, ncol=2)

    values = pd.concat([plot_df["best_dev_ppl"], plot_df["test_ppl"]])
    span = max(1e-6, values.max() - values.min())
    ymin = max(0, values.min() - max(2.0, 0.08 * span))
    ymax = values.max() + max(2.0, 0.14 * span)
    ax.set_ylim(ymin, ymax)

    for i, row in enumerate(plot_df.itertuples()):
        ax.text(i - width / 2, row.best_dev_ppl + (ymax - ymin) * 0.018, f"{row.best_dev_ppl:.1f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width / 2, row.test_ppl + (ymax - ymin) * 0.018, f"{row.test_ppl:.1f}", ha="center", va="bottom", fontsize=9)

    save_current_figure(figures_dir / "part1a_lr_search_absolute_ppl.png")


def plot_lr_search_delta(lr_df, figures_dir):
    plot_df = lr_df.sort_values("lr").copy()
    best = plot_df["best_dev_ppl"].min()
    plot_df["dev_delta_pct"] = (plot_df["best_dev_ppl"] - best) / best * 100
    plot_df["lr_label"] = plot_df["lr"].map(lambda x: f"{x:.0e}")
    best_idx = plot_df["dev_delta_pct"].idxmin()
    colors = [PAPER["orange"] if i == best_idx else PAPER["blue"] for i in plot_df.index]

    _, ax = plt.subplots(figsize=(9.8, 5.3))
    bars = ax.bar(plot_df["lr_label"], plot_df["dev_delta_pct"], color=colors, width=0.62)
    ax.axhline(0, color=PAPER["dark"], linewidth=1.0)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Dev PPL above best (%)")
    ax.set_title("Zoomed view of learning-rate sensitivity")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}%"))

    upper = max(1.0, plot_df["dev_delta_pct"].max() * 1.25)
    ax.set_ylim(0, upper)
    for bar, value in zip(bars, plot_df["dev_delta_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + upper * 0.025, f"{value:.2f}%", ha="center", va="bottom", fontsize=9)

    save_current_figure(figures_dir / "part1a_lr_search_zoomed_delta.png")


def plot_incremental_dev_test_ppl(results_df, figures_dir):
    plot_df = results_df.copy().reset_index(drop=True)
    plot_df["label"] = plot_df["name"].map(short_name)
    x = np.arange(len(plot_df))
    width = 0.36

    _, ax = plt.subplots(figsize=(12.5, 5.8))
    bars1 = ax.bar(x - width / 2, plot_df["best_dev_ppl"], width, color=PAPER["blue"], label="Dev PPL")
    bars2 = ax.bar(x + width / 2, plot_df["test_ppl"], width, color=PAPER["orange"], label="Test PPL")
    best_pos = int(plot_df["best_dev_ppl"].idxmin())
    ax.scatter([best_pos], [plot_df.loc[best_pos, "test_ppl"]], s=120, color=PAPER["red"], zorder=4, label="Selected by dev")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=28, ha="right")
    ax.set_ylabel("Perplexity")
    ax.set_title("Incremental Part 1.A experiment comparison")
    ax.legend(frameon=False, ncol=3)

    values = pd.concat([plot_df["best_dev_ppl"], plot_df["test_ppl"]])
    ax.set_ylim(max(0, values.min() - 3), values.max() + 7)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    save_current_figure(figures_dir / "part1a_incremental_dev_test_ppl.png")


def plot_test_ppl_ranking(results_df, figures_dir):
    plot_df = results_df.sort_values("test_ppl", ascending=True).copy()
    plot_df["label"] = plot_df["name"].map(short_name)
    best_name = results_df.sort_values("best_dev_ppl").iloc[0]["name"]
    colors = [PAPER["orange"] if name == best_name else PAPER["blue"] for name in plot_df["name"]]

    _, ax = plt.subplots(figsize=(9.2, 5.8))
    bars = ax.barh(plot_df["label"], plot_df["test_ppl"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Test perplexity")
    ax.set_title("Final test PPL ranking")
    xmin = max(0, plot_df["test_ppl"].min() - 4)
    xmax = plot_df["test_ppl"].max() + 6
    ax.set_xlim(xmin, xmax)
    for bar, value in zip(bars, plot_df["test_ppl"]):
        ax.text(value + 0.8, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", fontsize=9)

    save_current_figure(figures_dir / "part1a_test_ppl_ranking.png")


def plot_dev_ppl_curves(history_df, figures_dir):
    df = prepare_history(history_df)
    _, ax = plt.subplots(figsize=(12.3, 6.0))
    names = list(df["name"].drop_duplicates())
    color_cycle = [PAPER["blue"], PAPER["orange"], PAPER["green"], PAPER["red"], PAPER["purple"], PAPER["sky"], PAPER["gray"]]

    for idx, name in enumerate(names):
        sub = df[df["name"] == name]
        ax.plot(sub["epoch"], sub["dev_ppl"], marker="o", linewidth=2.0, markersize=4, color=color_cycle[idx % len(color_cycle)], label=short_name(name))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Development perplexity")
    ax.set_title("Development PPL curves across incremental experiments")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save_current_figure(figures_dir / "part1a_dev_ppl_curves.png")


def plot_best_train_dev_ppl_curve(history_df, best_name, figures_dir):
    df = prepare_history(history_df)
    sub = df[df["name"] == best_name].copy()
    _, ax = plt.subplots(figsize=(9.3, 5.4))

    ax.plot(sub["epoch"], sub["train_ppl"], marker="o", linewidth=2.3, color=PAPER["blue"], label="Train PPL")
    ax.plot(sub["epoch"], sub["dev_ppl"], marker="s", linewidth=2.3, color=PAPER["orange"], label="Dev PPL")

    best_epoch = int(sub.loc[sub["dev_ppl"].idxmin(), "epoch"])
    best_dev = float(sub["dev_ppl"].min())
    ax.scatter([best_epoch], [best_dev], color=PAPER["red"], s=115, zorder=5, label="Best dev epoch")
    ax.annotate(
        f"Best epoch {best_epoch}\nDev PPL {best_dev:.1f}",
        xy=(best_epoch, best_dev),
        xytext=(10, 14),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.8, color=PAPER["gray"]),
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity")
    ax.set_title(f"Train vs development PPL: {short_name(best_name)}")
    ax.legend(frameon=False)
    save_current_figure(figures_dir / "part1a_best_model_train_dev_ppl_curve.png")


def plot_best_loss_curve(history_df, best_name, figures_dir):
    sub = history_df[history_df["name"] == best_name].copy()
    _, ax = plt.subplots(figsize=(9.3, 5.4))

    ax.plot(sub["epoch"], sub["train_loss"], marker="o", linewidth=2.3, color=PAPER["blue"], label="Train loss")
    ax.plot(sub["epoch"], sub["dev_loss"], marker="s", linewidth=2.3, color=PAPER["orange"], label="Dev loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"Training loss dynamics: {short_name(best_name)}")
    ax.legend(frameon=False)
    save_current_figure(figures_dir / "part1a_best_model_loss_curve.png")


def plot_generalization_gap(history_df, best_name, figures_dir):
    df = prepare_history(history_df)
    sub = df[df["name"] == best_name].copy()
    _, ax = plt.subplots(figsize=(9.3, 5.4))

    ax.plot(sub["epoch"], sub["generalization_gap"], marker="o", linewidth=2.4, color=PAPER["purple"])
    ax.axhline(0, color=PAPER["dark"], linewidth=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dev PPL - Train PPL")
    ax.set_title(f"Generalization gap of selected model: {short_name(best_name)}")

    for _, row in sub.iterrows():
        ax.text(row["epoch"], row["generalization_gap"], f"{row['generalization_gap']:.1f}", ha="center", va="bottom", fontsize=8)

    save_current_figure(figures_dir / "part1a_generalization_gap.png")


def plot_relative_improvement(results_df, figures_dir):
    plot_df = results_df.copy().reset_index(drop=True)
    baseline_ppl = float(plot_df.loc[plot_df["name"] == "A0_baseline_chosen_lr", "test_ppl"].iloc[0])
    plot_df["improvement_pct"] = (baseline_ppl - plot_df["test_ppl"]) / baseline_ppl * 100
    plot_df["label"] = plot_df["name"].map(short_name)
    colors = [PAPER["orange"] if v == plot_df["improvement_pct"].max() else PAPER["green"] if v >= 0 else PAPER["red"] for v in plot_df["improvement_pct"]]

    _, ax = plt.subplots(figsize=(11.6, 5.5))
    bars = ax.bar(plot_df["label"], plot_df["improvement_pct"], color=colors)
    ax.axhline(0, color=PAPER["dark"], linewidth=1.0)
    ax.set_ylabel("Test PPL reduction vs baseline (%)")
    ax.set_title("Relative improvement over the selected baseline")
    ax.set_xticklabels(plot_df["label"], rotation=28, ha="right")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}%"))

    ymin = min(-1.0, plot_df["improvement_pct"].min() * 1.25)
    ymax = max(1.0, plot_df["improvement_pct"].max() * 1.25)
    ax.set_ylim(ymin, ymax)
    offset = (ymax - ymin) * 0.025

    for bar, value in zip(bars, plot_df["improvement_pct"]):
        y = value + offset if value >= 0 else value - offset
        va = "bottom" if value >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:.2f}%", ha="center", va=va, fontsize=9)

    save_current_figure(figures_dir / "part1a_relative_improvement.png")


def plot_parameter_tradeoff(results_df, figures_dir):
    plot_df = results_df.copy()
    plot_df["params_m"] = plot_df["total_params"] / 1e6

    _, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.scatter(plot_df["params_m"], plot_df["test_ppl"], s=95, color=PAPER["blue"], alpha=0.9)

    best_idx = plot_df["best_dev_ppl"].idxmin()
    ax.scatter([plot_df.loc[best_idx, "params_m"]], [plot_df.loc[best_idx, "test_ppl"]], s=160, color=PAPER["orange"], edgecolor=PAPER["dark"], linewidth=0.8, label="Selected")

    for _, row in plot_df.iterrows():
        ax.annotate(row["name"].split("_")[0], (row["params_m"], row["test_ppl"]), xytext=(6, 5), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("Test perplexity")
    ax.set_title("Model size versus test perplexity")
    ax.legend(frameon=False)
    save_current_figure(figures_dir / "part1a_params_vs_test_ppl.png")


def plot_best_epoch_comparison(results_df, full_experiment_epochs: int, figures_dir):
    plot_df = results_df.copy().reset_index(drop=True)
    plot_df["label"] = plot_df["name"].map(short_name)
    best_name = plot_df.sort_values("best_dev_ppl").iloc[0]["name"]
    colors = [PAPER["orange"] if name == best_name else PAPER["sky"] for name in plot_df["name"]]

    _, ax = plt.subplots(figsize=(10.8, 5.2))
    bars = ax.bar(plot_df["label"], plot_df["best_epoch"], color=colors)
    ax.set_ylabel("Selected epoch")
    ax.set_title("Best epoch selected by development PPL")
    ax.set_xticklabels(plot_df["label"], rotation=28, ha="right")
    ax.set_ylim(0, max(plot_df["best_epoch"].max() + 1, full_experiment_epochs + 1))

    for bar, value in zip(bars, plot_df["best_epoch"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.18, f"{int(value)}", ha="center", va="bottom", fontsize=9)

    save_current_figure(figures_dir / "part1a_best_epoch_comparison.png")


def plot_metric_heatmap(results_df, figures_dir):
    metric_df = results_df[["best_dev_ppl", "test_ppl", "best_epoch"]].copy()
    metric_df.index = results_df["name"].map(lambda x: x.split("_")[0])
    normalized = (metric_df - metric_df.min()) / (metric_df.max() - metric_df.min() + 1e-9)

    _, ax = plt.subplots(figsize=(7.8, 5.8))
    im = ax.imshow(normalized.values, aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(metric_df.shape[1]))
    ax.set_xticklabels(["Dev PPL", "Test PPL", "Best epoch"])
    ax.set_yticks(np.arange(metric_df.shape[0]))
    ax.set_yticklabels(metric_df.index)
    ax.set_title("Normalized summary of incremental experiments")

    for i in range(metric_df.shape[0]):
        for j in range(metric_df.shape[1]):
            value = metric_df.iloc[i, j]
            text = f"{value:.1f}" if j < 2 else f"{int(value)}"
            ax.text(j, i, text, ha="center", va="center", color="white" if normalized.iloc[i, j] > 0.45 else "black", fontsize=9)

    cbar = plt.gcf().colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized value")
    save_current_figure(figures_dir / "part1a_experiment_summary_heatmap.png")


def create_report_plots(lr_df, results_df, history_df, best_row, figures_dir, full_experiment_epochs: int):
    configure_matplotlib()
    plot_lr_search_absolute(lr_df, figures_dir)
    plot_lr_search_delta(lr_df, figures_dir)
    plot_incremental_dev_test_ppl(results_df, figures_dir)
    plot_test_ppl_ranking(results_df, figures_dir)
    plot_dev_ppl_curves(history_df, figures_dir)
    plot_best_train_dev_ppl_curve(history_df, best_row["name"], figures_dir)
    plot_best_loss_curve(history_df, best_row["name"], figures_dir)
    plot_generalization_gap(history_df, best_row["name"], figures_dir)
    plot_relative_improvement(results_df, figures_dir)
    plot_parameter_tradeoff(results_df, figures_dir)
    plot_best_epoch_comparison(results_df, full_experiment_epochs, figures_dir)
    plot_metric_heatmap(results_df, figures_dir)


def save_report_table(results_df, best_name: str, results_dir):
    summary_table = results_df[[
        "name", "lr", "d_model", "n_heads", "num_layers", "ff_dim", "dropout", "tie_weights",
        "best_epoch", "best_dev_ppl", "test_ppl", "total_params"
    ]].copy()
    summary_table["params_m"] = summary_table["total_params"] / 1e6
    summary_table = summary_table.drop(columns=["total_params"])
    out_path = results_dir / "part_A_report_table.csv"
    summary_table.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print(summary_table.to_string(index=False))
    print("\nSelected model:", best_name)
    return summary_table
