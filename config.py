"""Conservative default settings for the educational astronomy Mini-GPT."""

from dataclasses import asdict, dataclass


@dataclass
class GPTConfig:
    batch_size: int = 4
    block_size: int = 128
    embedding_size: int = 64
    number_of_heads: int = 4
    number_of_layers: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    number_of_steps: int = 20_000
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    eval_interval: int = 500
    eval_iters: int = 50
    train_split: float = 0.9
    seed: int = 1337
    deterministic: bool = False
    dataset_path: str = "data/input.txt"
    best_checkpoint_path: str = "models/astronomy_gpt_best.pt"
    latest_checkpoint_path: str = "models/astronomy_gpt_latest.pt"
    metrics_path: str = "logs/training_metrics.csv"
    temperature: float = 0.7
    top_k: int | None = 5
    top_p: float | None = None

    def to_dict(self):
        return asdict(self)
