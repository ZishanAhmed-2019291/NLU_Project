import os
import random
import urllib.request
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer


PTB_URLS = {
    "ptb.train.txt": "https://raw.githubusercontent.com/massimo-rizzoli/NLU-2026-Labs/main/labs/dataset/PennTreeBank/ptb.train.txt",
    "ptb.valid.txt": "https://raw.githubusercontent.com/massimo-rizzoli/NLU-2026-Labs/main/labs/dataset/PennTreeBank/ptb.valid.txt",
    "ptb.test.txt": "https://raw.githubusercontent.com/massimo-rizzoli/NLU-2026-Labs/main/labs/dataset/PennTreeBank/ptb.test.txt",
}


class PennTreeBank(Dataset):
    def __init__(self, corpus):
        self.sents = list(corpus)

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        return self.sents[idx]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_project_paths(project_dir: Path | None = None):
    project_dir = Path(project_dir) if project_dir is not None else Path(__file__).resolve().parent
    data_dir = project_dir / "dataset" / "PennTreeBank"
    results_dir = project_dir / "results"
    figures_dir = project_dir / "figures"
    bin_dir = project_dir / "bin"
    tmp_ckpt_dir = project_dir / "tmp_checkpoints_part1a"

    for path in [data_dir, results_dir, figures_dir, bin_dir, tmp_ckpt_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "project_dir": project_dir,
        "data_dir": data_dir,
        "results_dir": results_dir,
        "figures_dir": figures_dir,
        "bin_dir": bin_dir,
        "tmp_ckpt_dir": tmp_ckpt_dir,
    }


def download_ptb(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in PTB_URLS.items():
        out_path = data_dir / filename
        if not out_path.exists():
            print(f"Downloading {filename} ...")
            urllib.request.urlretrieve(url, out_path)
        else:
            print(f"Already exists: {filename}")


def read_file(path: Path, eos_token: str = "<eos>"):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line + " " + eos_token)
    return lines


def load_ptb_splits(data_dir: Path, download_if_missing: bool = True):
    required_files = ["ptb.train.txt", "ptb.valid.txt", "ptb.test.txt"]

    if download_if_missing and any(not (data_dir / filename).exists() for filename in required_files):
        download_ptb(data_dir)

    train_raw = read_file(data_dir / "ptb.train.txt")
    dev_raw = read_file(data_dir / "ptb.valid.txt")
    test_raw = read_file(data_dir / "ptb.test.txt")

    print("Train sentences:", len(train_raw))
    print("Dev sentences:", len(dev_raw))
    print("Test sentences:", len(test_raw))

    return train_raw, dev_raw, test_raw


def load_tokenizer(tokenizer_name: str = "openai-community/gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    print("Vocabulary size:", len(tokenizer))
    print("PAD token id:", tokenizer.pad_token_id)
    print("EOS token id:", tokenizer.eos_token_id)
    return tokenizer


def collate_fn(batch, tokenizer, block_size: int, device: str):
    tokenized = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=block_size + 1,
        return_tensors="pt",
    )
    input_ids = tokenized.input_ids[:, :-1].contiguous().to(device)
    labels = tokenized.input_ids[:, 1:].contiguous().to(device)
    n_tokens = (labels != tokenizer.pad_token_id).sum()
    return input_ids, labels, n_tokens


def make_dataloaders(
    train_raw,
    dev_raw,
    test_raw,
    tokenizer,
    device: str,
    batch_size_train: int,
    batch_size_eval: int,
    block_size: int,
    quick: bool = False,
    quick_train_size: int = 1200,
    quick_eval_size: int = 300,
):
    train_dataset = PennTreeBank(train_raw)
    dev_dataset = PennTreeBank(dev_raw)
    test_dataset = PennTreeBank(test_raw)

    if quick:
        train_dataset = Subset(train_dataset, range(min(quick_train_size, len(train_dataset))))
        dev_dataset = Subset(dev_dataset, range(min(quick_eval_size, len(dev_dataset))))
        test_dataset = Subset(test_dataset, range(min(quick_eval_size, len(test_dataset))))

    collate = partial(collate_fn, tokenizer=tokenizer, block_size=block_size, device=device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size_eval, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_eval, shuffle=False, collate_fn=collate)
    return train_loader, dev_loader, test_loader
