import numpy as np
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

    def _init_weights(self) -> None:
        """Initialises all ``nn.Linear`` weights with He (Kaiming) uniform init.

        Biases are initialised to zero. Called automatically at the end of
        ``__init__``.
        """
        # He uniform initialisation for all Linear layers.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch_size, input_dim)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(batch_size, 1)`` with
            values in ``[0, 1]`` representing the probability of the
            malicious class.
        """
        return self.network(x)
    

class PacketsDataset(Dataset):
    """PyTorch ``Dataset`` wrapping pre-processed packet feature arrays.

    Converts numpy arrays to float32 tensors at construction time so that
    per-batch device transfers are the only copy required during training.

    Args:
        X (np.ndarray): Feature matrix of shape ``(n_samples, n_features)``.
        y (np.ndarray): Binary label vector of shape ``(n_samples,)``.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initialises the dataset by converting arrays to float32 tensors.

        Args:
            X (np.ndarray): Feature matrix of shape ``(n_samples, n_features)``.
            y (np.ndarray): Binary label vector of shape ``(n_samples,)``.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves a single sample by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A ``(features, label)`` pair
            where features has shape ``(n_features,)`` and label has
            shape ``(1,)``.
        """
        return self.X[idx], self.y[idx]

