"""
MicroBiConvLSTM: An Ultra-Lightweight Bidirectional Convolutional LSTM for Human Activity Recognition.

Architecture: Conv1D -> Pool -> Conv1D -> Pool -> BiLSTM -> Aggregation -> Classifier.
Frozen configuration yields approximately 10,454 parameters for UCI-HAR.
Designed for efficient inference on edge devices with O(N) complexity.

Author: Mridankan Mandal
Paper: https://arxiv.org/abs/2602.06523
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


# Frozen architecture configuration. Only training hyperparameters (lr, weightDecay, dropout)
# are tuned via HPO. These values are locked for the research paper.
MICRO_BI_CONV_LSTM_CONFIG = {
    'convFilters': 16,
    'convKernel': 5,
    'convPadding': 2,
    'numConvLayers': 2,
    'poolSize': 2,
    'poolStride': 2,
    'lstmHidden': 24,
    'lstmLayers': 1,
    'bidirectional': True,
    'lstmOutput': 48,
}


class MicroBiConvLSTM(nn.Module):
    """Ultra-lightweight convolutional BiLSTM for human activity recognition.

    Stages:
        I.   Conv1D Stem:    inChannels -> convFilters, k=5, BatchNorm, ReLU, MaxPool.
        II.  Conv1D Block:   convFilters -> convFilters, k=5, BatchNorm, ReLU, MaxPool.
        III. BiLSTM:         convFilters -> lstmHidden * 2 (bidirectional).
        IV.  Aggregation:    Last timestep or mean pooling.
        V.   Classifier:     Linear(lstmOutput, numClasses).

    Args:
        numClasses:   Number of activity classes.
        inChannels:   Number of input sensor channels.
        seqLen:       Sequence length in timesteps.
        dropout:      Dropout rate before the classifier head.
        aggregation:  Temporal aggregation method ('last' or 'mean').
        convFilters:  Override conv filter count (for ablation studies only).
        lstmHidden:   Override LSTM hidden size (for ablation studies only).
    """

    def __init__(
        self,
        numClasses: int = 6,
        inChannels: int = 9,
        seqLen: int = 128,
        dropout: float = 0.1,
        aggregation: str = 'last',
        convFilters: Optional[int] = None,
        lstmHidden: Optional[int] = None,
    ):
        super().__init__()

        self.convFilters = convFilters or MICRO_BI_CONV_LSTM_CONFIG['convFilters']
        self.convKernel = MICRO_BI_CONV_LSTM_CONFIG['convKernel']
        self.convPadding = MICRO_BI_CONV_LSTM_CONFIG['convPadding']
        self.poolSize = MICRO_BI_CONV_LSTM_CONFIG['poolSize']
        self.lstmHidden = lstmHidden or MICRO_BI_CONV_LSTM_CONFIG['lstmHidden']
        self.lstmLayers = MICRO_BI_CONV_LSTM_CONFIG['lstmLayers']
        self.bidirectional = MICRO_BI_CONV_LSTM_CONFIG['bidirectional']

        self.numClasses = numClasses
        self.inChannels = inChannels
        self.seqLen = seqLen
        self.dropout = dropout
        self.aggregation = aggregation

        self.seqLenAfterPool = seqLen // 4
        self.lstmOutput = self.lstmHidden * 2 if self.bidirectional else self.lstmHidden

        # Stage I: Convolutional Stem.
        self.conv1 = nn.Conv1d(
            inChannels, self.convFilters,
            kernel_size=self.convKernel, padding=self.convPadding, stride=1, bias=True,
        )
        self.bn1 = nn.BatchNorm1d(self.convFilters)
        self.pool1 = nn.MaxPool1d(kernel_size=self.poolSize, stride=self.poolSize)

        # Stage II: Second Convolutional Block.
        self.conv2 = nn.Conv1d(
            self.convFilters, self.convFilters,
            kernel_size=self.convKernel, padding=self.convPadding, stride=1, bias=True,
        )
        self.bn2 = nn.BatchNorm1d(self.convFilters)
        self.pool2 = nn.MaxPool1d(kernel_size=self.poolSize, stride=self.poolSize)

        # Stage III: Bidirectional LSTM.
        self.lstm = nn.LSTM(
            input_size=self.convFilters,
            hidden_size=self.lstmHidden,
            num_layers=self.lstmLayers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        # Stage V: Classification Head.
        self.dropoutLayer = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.lstmOutput, numClasses)

        self._initWeights()

    def _initWeights(self):
        """Initialize weights with best-practice defaults."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                        n = param.size(0)
                        param.data[n // 4:n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input: [B, T, C]. Output: [B, numClasses]."""
        x = x.permute(0, 2, 1)                    # [B, C, T]

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.pool1(x)                          # [B, F, T/2]

        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.pool2(x)                          # [B, F, T/4]

        x = x.permute(0, 2, 1)                    # [B, T/4, F]
        x, _ = self.lstm(x)                        # [B, T/4, lstmOutput]

        if self.aggregation == 'last':
            x = x[:, -1, :]
        else:
            x = x.mean(dim=1)

        x = self.dropoutLayer(x)
        return self.classifier(x)

    def countParameters(self) -> Dict[str, int]:
        """Count parameters grouped by stage."""
        return {
            'total': sum(p.numel() for p in self.parameters()),
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'convStem': sum(p.numel() for p in self.conv1.parameters()) + sum(p.numel() for p in self.bn1.parameters()),
            'convBlock': sum(p.numel() for p in self.conv2.parameters()) + sum(p.numel() for p in self.bn2.parameters()),
            'biLstm': sum(p.numel() for p in self.lstm.parameters()),
            'classifier': sum(p.numel() for p in self.classifier.parameters()),
        }

    def getModelInfo(self) -> Dict[str, Any]:
        """Return a summary dictionary for logging."""
        params = self.countParameters()
        return {
            'modelName': 'MicroBiConvLSTM',
            'numClasses': self.numClasses,
            'inChannels': self.inChannels,
            'seqLen': self.seqLen,
            'convFilters': self.convFilters,
            'lstmHidden': self.lstmHidden,
            'bidirectional': self.bidirectional,
            'aggregation': self.aggregation,
            'dropout': self.dropout,
            'totalParameters': params['total'],
            'trainableParameters': params['trainable'],
        }


# Dataset-specific input configurations.
DATASET_CONFIGS = {
    'ucihar': {'inChannels': 9, 'seqLen': 128, 'numClasses': 6},
    'motionsense': {'inChannels': 6, 'seqLen': 128, 'numClasses': 6},
    'wisdm': {'inChannels': 3, 'seqLen': 128, 'numClasses': 6},
    'pamap2': {'inChannels': 19, 'seqLen': 128, 'numClasses': 12},
    'opportunity': {'inChannels': 79, 'seqLen': 128, 'numClasses': 5},
    'unimib': {'inChannels': 3, 'seqLen': 128, 'numClasses': 9},
    'skoda': {'inChannels': 30, 'seqLen': 98, 'numClasses': 11},
    'daphnet': {'inChannels': 9, 'seqLen': 64, 'numClasses': 2},
}


def createMicroBiConvLstm(
    datasetName: str,
    dropout: float = 0.1,
    aggregation: str = 'last',
) -> MicroBiConvLSTM:
    """Factory function to create a MicroBiConvLSTM configured for a specific dataset.

    Args:
        datasetName:  Name of the dataset (e.g. 'ucihar', 'pamap2').
        dropout:      Dropout rate for the classifier head.
        aggregation:  Temporal aggregation method ('last' or 'mean').

    Returns:
        A configured MicroBiConvLSTM instance.
    """
    dsName = datasetName.lower().replace('-', '').replace('_', '')

    if dsName not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {datasetName}. Available: {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[dsName]
    return MicroBiConvLSTM(
        numClasses=config['numClasses'],
        inChannels=config['inChannels'],
        seqLen=config['seqLen'],
        dropout=dropout,
        aggregation=aggregation,
    )


if __name__ == '__main__':
    model = createMicroBiConvLstm('ucihar', dropout=0.1)

    info = model.getModelInfo()
    print(f"\n{'=' * 60}")
    print("MicroBiConvLSTM Model Summary")
    print(f"{'=' * 60}")
    for key, value in info.items():
        print(f"  {key}: {value}")

    x = torch.randn(32, 128, 9)
    y = model(x)
    print(f"\n  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")

    params = model.countParameters()
    print(f"\n{'=' * 60}")
    print("Parameter Breakdown")
    print(f"{'=' * 60}")
    for key, value in params.items():
        print(f"  {key}: {value:,}")

    print("\nMicroBiConvLSTM ready for training.")
