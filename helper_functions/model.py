import torch
import torch.nn as nn
from torch.utils.data import Dataset

class NetworkAnomalyDetector(nn.Module):
    """
    Fully-connected binary classifier for network anomaly detection.

    Architecture:
        Input  -> Linear(256) -> BN -> ReLU -> Dropout
               -> Linear(128) -> BN -> ReLU -> Dropout
               -> Linear(64)  -> BN -> ReLU -> Dropout
               -> Linear(32)  -> BN -> ReLU -> Dropout
               -> Linear(1)   -> Sigmoid

    Args:
        input_dim (int):   Number of features after preprocessing.
        dropout_p (float): Dropout probability for each hidden block.
    """

    def __init__(self, input_dim: int, dropout_p: float = 0.3):
        super().__init__()

        self.network = nn.Sequential(
            # Block 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            # Block 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            # Block 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            # Block 4
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            # Output
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        # He uniform initialisation for all Linear layers.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    

class PacketsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

