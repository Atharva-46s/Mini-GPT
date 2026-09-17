"""Dataset loading and random batch creation."""

from pathlib import Path
import torch


def load_dataset(path, tokenizer, train_split, block_size):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    text = path.read_text(encoding="utf-8")
    if len(text) < 2 * (block_size + 1):
        raise ValueError("Dataset is too small for the selected block_size and validation split.")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_index = int(train_split * len(data))
    if split_index <= block_size or len(data) - split_index <= block_size:
        raise ValueError("Train or validation split is too small for block_size.")
    stats = {"characters": len(text), "words": len(text.split()), "vocab_size": tokenizer.vocab_size,
             "train_characters": split_index, "validation_characters": len(data) - split_index}
    return data[:split_index], data[split_index:], stats


def get_batch(split, train_data, validation_data, batch_size, block_size, device):
    selected_data = train_data if split == "train" else validation_data
    starts = torch.randint(0, len(selected_data) - block_size, (batch_size,))
    x = torch.stack([selected_data[start:start + block_size] for start in starts])
    y = torch.stack([selected_data[start + 1:start + block_size + 1] for start in starts])
    return x.to(device), y.to(device)
