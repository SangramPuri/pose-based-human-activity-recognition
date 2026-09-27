import torch
from src.model import ActivityLSTM

def test_model_output_shape():
    model=ActivityLSTM(); output=model(torch.randn(4,32,34)); assert output.shape==(4,6)
