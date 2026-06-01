import torch
import torch.nn as nn

class FastPolicyNet(nn.Module):
    """Policy Network Ringan (Shallow) untuk komputasi cepat Rollout MCTS."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(31, 32)   # Input: 31 fitur state -> Hidden Layer: 32 neuron
        self.fc2 = nn.Linear(32, 3)    # Output: 3 pilihan aksi legal (ambil 1, 2, 3 koin)

    def forward(self, x):
        x = torch.relu(self.fc1(x))    # Aktivasi non-linear ReLU
        return torch.softmax(self.fc2(x), dim=1)  # Distribusi probabilitas aksi

class StrongPolicyNet(nn.Module):
    """Policy Network Dalam (Deep) untuk mempelajari strategi kompleks."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(31, 64)   # Input Layer ke Hidden 1 (64 neuron)
        self.fc2 = nn.Linear(64, 32)   # Hidden 1 ke Hidden 2 (32 neuron)
        self.fc3 = nn.Linear(32, 16)   # Hidden 2 ke Hidden 3 (16 neuron)
        self.fc4 = nn.Linear(16, 3)    # Output Layer (3 pilihan aksi)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.softmax(self.fc4(x), dim=1)

class PolicyNet(nn.Module):
    """Arsitektur Policy Network standar dioptimalkan dengan Policy Gradient."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(31, 32)   # Hidden Layer 1 (32 neuron)
        self.fc2 = nn.Linear(32, 32)   # Hidden Layer 2 (32 neuron)
        self.fc3 = nn.Linear(32, 3)    # Output Layer (3 pilihan aksi)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class ValueNet(nn.Module):
    """Value Network untuk evaluasi probabilitas biner hasil akhir state."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(31, 64)   # Hidden Layer 1 (64 neuron)
        self.fc2 = nn.Linear(64, 32)   # Hidden Layer 2 (32 neuron)
        self.fc3 = nn.Linear(32, 16)   # Hidden Layer 3 (16 neuron)
        self.fc4 = nn.Linear(16, 2)    # Output: Klasifikasi biner [Peluang Kalah, Peluang Menang]

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.softmax(self.fc4(x), dim=-1)