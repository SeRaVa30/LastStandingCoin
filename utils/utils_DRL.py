import numpy as np
import torch

def onehot_encoder(state):
    """Mengonversi integer state menjadi tensor One-Hot 1x31 untuk input model."""
    onehot = np.zeros((1, 31))
    onehot[0, state] = 1
    return torch.FloatTensor(onehot)