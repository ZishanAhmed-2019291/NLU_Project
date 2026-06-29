# LM Part 1.A - Scratch Transformer Language Modeling

This folder contains the code for Part 1.A of the Language Modeling task. The model is a compact GPT-style decoder-only Transformer trained from scratch on the Penn Treebank dataset.

## Files

- `main.py`: runs the complete experiment pipeline.
- `model.py`: defines the scratch Transformer language model.
- `utils.py`: handles dataset loading, tokenization, dataloaders, paths, and reproducibility utilities.
- `functions.py`: contains training, evaluation, checkpointing, result-table generation, and plotting functions.
- `dataset/`: contains the Penn Treebank train, development, and test files.
- `bin/`: contains the best trained model binary.

## How to run

From inside this folder, run:

```bash
python main.py
```

The script performs:

1. Penn Treebank loading and GPT-2 tokenizer setup.
2. Learning-rate search for the scratch baseline.
3. Incremental architecture experiments.
4. Best model selection using development perplexity.
5. Final test evaluation of the selected model.
6. Saving result tables and report figures.
7. Saving the selected best model in `bin/`.

## Outputs

The run creates or updates:

- `results/part1a_lr_search_results.csv`
- `results/part1a_incremental_results.csv`
- `results/part1a_history.csv`
- `results/part1a_best_model_metadata.json`
- `figures/*.png`
- `bin/part1a_best_model.pt`

Only the best model selected by development perplexity is saved in the final `bin/` folder.

## Main experiment settings

The final run uses:

- learning-rate search over multiple values
- incremental architecture changes
- development perplexity for model selection
- test perplexity only for final held-out evaluation
- token-level cross-entropy with padding ignored

The selected model and all final metrics are written to the results folder after execution.
