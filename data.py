from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset

def convert_frame(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size != 36:
        raise ValueError(f"Expected 36 values per frame, received {arr.size}.")
    return np.concatenate([arr[:2], arr[4:]])

def read_pose_file(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            values = [float(x.strip()) for x in line.split(",")]
            try:
                rows.append(convert_frame(values))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not rows:
        raise ValueError(f"No pose records found in {path}.")
    return np.asarray(rows, dtype=np.float32)

def read_labels(path: Path):
    labels = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            labels.extend(int(x) - 1 for x in line.replace(",", " ").split())
    if not labels:
        raise ValueError(f"No labels found in {path}.")
    return np.asarray(labels, dtype=np.int64)

def build_sequences(frames, labels, sequence_length=32):
    if len(frames) % sequence_length:
        raise ValueError("Frame count must be divisible by sequence length.")
    n = len(frames) // sequence_length
    X = frames.reshape(n, sequence_length, frames.shape[1])
    if len(labels) != n:
        raise ValueError(f"Labels ({len(labels)}) do not match sequences ({n}).")
    return X, labels

class PoseSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, index): return self.X[index], self.y[index]

def load_split(data_dir: Path, split: str, sequence_length=32):
    return build_sequences(
        read_pose_file(data_dir / f"X_{split}.txt"),
        read_labels(data_dir / f"Y_{split}.txt"),
        sequence_length,
    )

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
