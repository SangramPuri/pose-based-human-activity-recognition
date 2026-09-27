import torch
from torch import nn

class ActivityLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_layers=2, num_classes=6, dropout=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        encoded = self.encoder(x)
        sequence_output, _ = self.lstm(encoded)
        return self.classifier(sequence_output[:, -1, :])
