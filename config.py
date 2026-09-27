from dataclasses import dataclass
from pathlib import Path

LABELS = [
    "JUMPING", "JUMPING_JACKS", "BOXING",
    "WAVING_2HANDS", "WAVING_1HAND", "CLAPPING_HANDS",
]

@dataclass
class Config:
    sequence_length: int = 32
    input_features: int = 34
    hidden_size: int = 64
    num_layers: int = 2
    num_classes: int = 6
    dropout: float = 0.25
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 40
    seed: int = 42
    data_dir: Path = Path("data/RNN-HAR-2D-Pose-database")
    model_dir: Path = Path("models")
