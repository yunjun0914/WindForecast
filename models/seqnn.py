import torch
from torch import nn
import torch.nn.functional as F


class GRUPowerRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.10):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, max(16, hidden_size // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, hidden_size // 2), 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class DLinearPowerRegressor(nn.Module):
    def __init__(self, input_size, window, hidden_size=64, dropout=0.10):
        super().__init__()
        self.window = window
        self.input_size = input_size
        self.trend_head = nn.Linear(window * input_size, 1)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(window * input_size),
            nn.Linear(window * input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        trend = x.mean(dim=1, keepdim=True)
        seasonal = x - trend
        trend_flat = trend.repeat(1, self.window, 1).reshape(x.shape[0], -1)
        seasonal_flat = seasonal.reshape(x.shape[0], -1)
        return (self.trend_head(trend_flat) + self.residual_head(seasonal_flat)).squeeze(-1)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return F.relu(out + residual)


class TCNPowerRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=4, kernel_size=3, dropout=0.10):
        super().__init__()
        layers = []
        in_channels = input_size
        for layer_idx in range(num_layers):
            dilation = 2**layer_idx
            layers.append(TemporalBlock(in_channels, hidden_size, kernel_size, dilation, dropout))
            in_channels = hidden_size
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, max(16, hidden_size // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, hidden_size // 2), 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)
        last = out[:, :, -1]
        return self.head(last).squeeze(-1)


class TCNRegimeClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        n_classes=4,
        hidden_size=64,
        num_layers=1,
        kernel_size=3,
        dropout=0.10,
    ):
        super().__init__()
        layers = []
        in_channels = input_size
        for layer_idx in range(num_layers):
            dilation = 2**layer_idx
            layers.append(
                TemporalBlock(
                    in_channels,
                    hidden_size,
                    kernel_size,
                    dilation,
                    dropout,
                )
            )
            in_channels = hidden_size
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, max(16, hidden_size // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, hidden_size // 2), n_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)
        return self.head(out[:, :, -1])
