import json
from pathlib import Path

import pandas as pd

from functions import ExperimentConfig, run_experiments, save_best_model, create_report_plots, save_report_table
from utils import set_seed, get_device, get_project_paths, load_ptb_splits, load_tokenizer


SEED = 42
RUN_MODE = "final"
QUICK = RUN_MODE == "quick"

LR_SEARCH_EPOCHS = 4
FULL_EXPERIMENT_EPOCHS = 5
PATIENCE = 4


def build_lr_search_experiments():
    experiments = [
        ExperimentConfig(name="A0_baseline_lr_1e-3", lr=1e-3, epochs=LR_SEARCH_EPOCHS, patience=PATIENCE),
        ExperimentConfig(name="A0_baseline_lr_5e-4", lr=5e-4, epochs=LR_SEARCH_EPOCHS, patience=PATIENCE),
        ExperimentConfig(name="A0_baseline_lr_3e-4", lr=3e-4, epochs=LR_SEARCH_EPOCHS, patience=PATIENCE),
        ExperimentConfig(name="A0_baseline_lr_1e-4", lr=1e-4, epochs=LR_SEARCH_EPOCHS, patience=PATIENCE),
    ]

    if QUICK:
        experiments = [
            ExperimentConfig(
                name="quick_baseline",
                lr=5e-4,
                d_model=64,
                n_heads=2,
                num_layers=1,
                ff_dim=128,
                epochs=2,
                patience=2,
                batch_size_train=16,
                batch_size_eval=32,
                block_size=64,
            )
        ]

    return experiments


def build_incremental_experiments(chosen_lr: float):
    experiments = [
        ExperimentConfig(
            name="A0_baseline_chosen_lr",
            lr=chosen_lr,
            d_model=128,
            n_heads=2,
            num_layers=2,
            ff_dim=512,
            dropout=0.0,
            tie_weights=False,
            epochs=FULL_EXPERIMENT_EPOCHS,
            patience=PATIENCE,
        ),
        ExperimentConfig(
            name="A1_change_d_model_192",
            lr=chosen_lr,
            d_model=192,
            n_heads=2,
            num_layers=2,
            ff_dim=512,
            dropout=0.0,
            tie_weights=False,
            epochs=FULL_EXPERIMENT_EPOCHS,
            patience=PATIENCE,
        ),
        ExperimentConfig(
            name="A2_change_n_heads_4",
            lr=chosen_lr,
            d_model=192,
            n_heads=4,
            num_layers=2,
            ff_dim=512,
            dropout=0.0,
            tie_weights=False,
            epochs=FULL_EXPERIMENT_EPOCHS,
            patience=PATIENCE,
        ),
        ExperimentConfig(
            name="A3_change_num_layers_3",
            lr=chosen_lr,
            d_model=192,
            n_heads=4,
            num_layers=3,
            ff_dim=512,
            dropout=0.0,
            tie_weights=False,
            epochs=FULL_EXPERIMENT_EPOCHS,
            patience=PATIENCE,
        ),
        ExperimentConfig(
            name="A4_change_ff_dim_768",
            lr=chosen_lr,
            d_model=192,
            n_heads=4,
            num_layers=3,
            ff_dim=768,
            dropout=0.0,
            tie_weights=False,
            epochs=FULL_EXPERIMENT_EPOCHS,
            patience=PATIENCE,
        ),
        ExperimentConfig(
            name="A5_add_dropout_0_1",
            lr=chosen_lr,
            d_model=192,
            n_heads=4,
            num_layers=3,
            ff_dim=768,
            dropout=0.1,
            tie_weights=False,
            epochs=FULL_EXPERIMENT_EPOCHS,
            patience=PATIENCE,
        ),
        ExperimentConfig(
            name="A6_add_weight_tying",
            lr=chosen_lr,
            d_model=192,
            n_heads=4,
            num_layers=3,
            ff_dim=768,
            dropout=0.1,
            tie_weights=True,
            epochs=FULL_EXPERIMENT_EPOCHS,
            patience=PATIENCE,
        ),
    ]

    if QUICK:
        experiments = [
            ExperimentConfig(
                name="quick_dropout_weight_tying",
                lr=chosen_lr,
                d_model=64,
                n_heads=2,
                num_layers=1,
                ff_dim=128,
                dropout=0.1,
                tie_weights=True,
                epochs=2,
                patience=2,
                batch_size_train=16,
                batch_size_eval=32,
                block_size=64,
            )
        ]

    return experiments


def main():
    set_seed(SEED)
    device = get_device()
    paths = get_project_paths(Path(__file__).resolve().parent)

    print("Device:", device)
    print("Project directory:", paths["project_dir"])
    print("Run mode:", RUN_MODE)
    print("LR-search epochs:", LR_SEARCH_EPOCHS)
    print("Full experiment epochs:", FULL_EXPERIMENT_EPOCHS)

    splits = load_ptb_splits(paths["data_dir"])
    tokenizer = load_tokenizer("openai-community/gpt2")

    lr_experiments = build_lr_search_experiments()
    lr_df, lr_history_df = run_experiments(lr_experiments, tokenizer, splits, paths, device, quick=QUICK)
    lr_df = lr_df.sort_values("best_dev_ppl").reset_index(drop=True)

    lr_csv_path = paths["results_dir"] / "part_A_baseline_lr_search.csv"
    lr_history_csv_path = paths["results_dir"] / "part_A_lr_search_histories.csv"
    lr_df.to_csv(lr_csv_path, index=False)
    lr_history_df.to_csv(lr_history_csv_path, index=False)

    chosen_lr = float(lr_df.iloc[0]["lr"])
    print("Saved:", lr_csv_path)
    print("Saved:", lr_history_csv_path)
    print("Chosen LR:", chosen_lr)
    print(lr_df[["name", "lr", "best_epoch", "best_dev_ppl", "test_ppl", "total_params"]].to_string(index=False))

    incremental_experiments = build_incremental_experiments(chosen_lr)
    results_df, history_df = run_experiments(incremental_experiments, tokenizer, splits, paths, device, quick=QUICK)

    results_csv_path = paths["results_dir"] / "part_A_incremental_results.csv"
    history_csv_path = paths["results_dir"] / "part_A_incremental_histories.csv"
    results_df.to_csv(results_csv_path, index=False)
    history_df.to_csv(history_csv_path, index=False)

    print("Saved:", results_csv_path)
    print("Saved:", history_csv_path)
    print(results_df[[
        "name", "lr", "d_model", "n_heads", "num_layers", "ff_dim", "dropout", "tie_weights",
        "best_epoch", "best_dev_ppl", "test_ppl", "total_params"
    ]].to_string(index=False))

    best_row, final_model_path, metadata_path = save_best_model(results_df, paths)
    create_report_plots(lr_df, results_df, history_df, best_row, paths["figures_dir"], FULL_EXPERIMENT_EPOCHS)
    save_report_table(results_df, best_row["name"], paths["results_dir"])

    print("\nFinal best model:", final_model_path)
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
