"""
DeepConvLSTM Baseline Implementation
Deep Convolutional and LSTM Recurrent Neural Networks for HAR

Reference: Ordóñez & Roggen, "Deep Convolutional and LSTM RNNs for HAR," Sensors 2016
Original: ~300k parameters

This is a simplified reimplementation for fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepConvLSTM(nn.Module):
    """
    DeepConvLSTM: Classic Deep Learning HAR Model.
    
    Architecture (per Ordóñez & Roggen, Sensors 2016):
    - 4 convolutional layers
    - 2 LSTM layers (unidirectional by default, as in original paper)
    - Dense classification head
    
    Args:
        numClasses: Number of activity classes
        inChannels: Number of input sensor channels
        seqLen: Sequence length (timesteps)
        convFilters: Number of filters per conv layer
        lstmHidden: Hidden size for LSTM
        lstmLayers: Number of LSTM layers
        dropout: Dropout rate
        bidirectional: Whether to use bidirectional LSTM
        
    Reference: Ordóñez & Roggen, "Deep Convolutional and LSTM RNNs for HAR," Sensors 2016
    
    Default config (~132K params, unidirectional as per original paper):
        convFilters=64, lstmHidden=64, lstmLayers=2, bidirectional=False
    """
    
    def __init__(
        self,
        numClasses: int = 6,
        inChannels: int = 9,
        seqLen: int = 128,
        convFilters: int = 64,
        lstmHidden: int = 64,
        lstmLayers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = False,
    ):
        super().__init__()
        
        self.numClasses = numClasses
        self.inChannels = inChannels
        self.seqLen = seqLen
        self.bidirectional = bidirectional
        
        # Convolutional layers (4 layers as in original)
        self.convLayers = nn.Sequential(
            # Conv1
            nn.Conv1d(inChannels, convFilters, kernel_size=5, padding=2),
            nn.BatchNorm1d(convFilters),
            nn.ReLU(inplace=True),
            
            # Conv2
            nn.Conv1d(convFilters, convFilters, kernel_size=5, padding=2),
            nn.BatchNorm1d(convFilters),
            nn.ReLU(inplace=True),
            
            # Conv3
            nn.Conv1d(convFilters, convFilters, kernel_size=5, padding=2),
            nn.BatchNorm1d(convFilters),
            nn.ReLU(inplace=True),
            
            # Conv4
            nn.Conv1d(convFilters, convFilters, kernel_size=5, padding=2),
            nn.BatchNorm1d(convFilters),
            nn.ReLU(inplace=True),
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=convFilters,
            hidden_size=lstmHidden,
            num_layers=lstmLayers,
            batch_first=True,
            dropout=dropout if lstmLayers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Classification head
        lstmOutputSize = lstmHidden * 2 if bidirectional else lstmHidden
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstmOutputSize, numClasses),
        )
        
        self._initWeights()
        
    def _initWeights(self):
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
                    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, T, C] or [B, C, T]
            
        Returns:
            Logits [B, numClasses]
        """
        # Ensure [B, C, T] format for convolutions
        if x.dim() == 3 and x.shape[-1] != self.inChannels:
            # Already [B, C, T]
            pass
        else:
            # [B, T, C] -> [B, C, T]
            x = x.permute(0, 2, 1)
            
        # Convolutional layers
        x = self.convLayers(x)
        
        # LSTM expects [B, T, C]
        x = x.permute(0, 2, 1)
        x, (hn, cn) = self.lstm(x)
        
        # Use last hidden state (handle bidirectional)
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            x = torch.cat([hn[-2], hn[-1]], dim=-1)  # [B, lstmHidden*2]
        else:
            x = hn[-1]  # [B, lstmHidden]
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def countParameters(self) -> dict:
        """Count parameters by component."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'convLayers': sum(p.numel() for p in self.convLayers.parameters()),
            'lstm': sum(p.numel() for p in self.lstm.parameters()),
            'classifier': sum(p.numel() for p in self.classifier.parameters()),
        }


def createDeepConvLstm(dataset: str = 'ucihar') -> DeepConvLSTM:
    """
    Factory function for dataset-specific DeepConvLSTM models.
    
    Args:
        dataset: Dataset name
        
    Returns:
        Configured DeepConvLSTM model
    """
    configs = {
        'ucihar': {'numClasses': 6, 'inChannels': 9, 'seqLen': 128},
        'motionsense': {'numClasses': 6, 'inChannels': 6, 'seqLen': 128},
        'wisdm': {'numClasses': 6, 'inChannels': 3, 'seqLen': 128},
        'pamap2': {'numClasses': 12, 'inChannels': 19, 'seqLen': 128},
        'opportunity': {'numClasses': 5, 'inChannels': 79, 'seqLen': 128},
        'unimib': {'numClasses': 9, 'inChannels': 3, 'seqLen': 128},
        'skoda': {'numClasses': 11, 'inChannels': 30, 'seqLen': 98},
        'daphnet': {'numClasses': 2, 'inChannels': 9, 'seqLen': 64},
    }
    
    if dataset.lower() not in configs:
        raise ValueError(f"Unknown dataset: {dataset}")
        
    cfg = configs[dataset.lower()]
    return DeepConvLSTM(**cfg)


if __name__ == '__main__':
    # Quick test
    print("=== DeepConvLSTM (~132K, unidirectional per original paper) ===")
    model = DeepConvLSTM(numClasses=6, inChannels=9, seqLen=128)
    params = model.countParameters()
    
    for k, v in params.items():
        print(f"  {k}: {v:,}")
        
    # Test forward pass
    x = torch.randn(2, 128, 9)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
