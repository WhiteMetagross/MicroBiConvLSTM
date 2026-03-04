"""
TinierHAR Baseline Implementation
Ultra-lightweight HAR model with depthwise separable convolutions

Reference: "TinierHAR: Towards Ultra-Lightweight Deep Learning Models 
           for Efficient HAR on Edge Devices" (UbiComp/ISWC 2025)
GitHub: zhaxidele/TinierHAR

Target: ~17k parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable 2D convolution (same as TinierHAR repo)."""
    
    def __init__(self, inChannels: int, outChannels: int, kernelSize: int = 5):
        super().__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            inChannels, inChannels, 
            kernel_size=(kernelSize, 1), 
            padding=(kernelSize // 2, 0), 
            groups=inChannels,
            bias=False
        )
        # Pointwise convolution
        self.pointwise = nn.Conv2d(inChannels, outChannels, kernel_size=1, bias=False)
        
    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class TinierHARConvBlock(nn.Module):
    """
    ConvBlock with residual shortcut (matches TinierHAR repo exactly).
    
    Uses depthwise separable conv + BN + ReLU + optional MaxPool + shortcut.
    """
    
    def __init__(
        self, 
        inChannels: int, 
        outChannels: int, 
        kernelSize: int = 5,
        useMaxpool: bool = True, 
        useShortcut: bool = True
    ):
        super().__init__()
        self.useMaxpool = useMaxpool
        self.useShortcut = useShortcut
        
        # Main path: DW-Sep Conv -> BN -> ReLU -> (MaxPool)
        layers = [
            DepthwiseSeparableConv2d(inChannels, outChannels, kernelSize),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True)
        ]
        if useMaxpool:
            layers.append(nn.MaxPool2d((2, 1)))
        self.main = nn.Sequential(*layers)
        
        # Shortcut path
        if useShortcut:
            shortcutLayers = []
            if inChannels != outChannels:
                shortcutLayers.extend([
                    nn.Conv2d(inChannels, outChannels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(outChannels)
                ])
            if useMaxpool:
                shortcutLayers.append(nn.MaxPool2d((2, 1)))
            self.shortcut = nn.Sequential(*shortcutLayers) if shortcutLayers else nn.Identity()
        else:
            self.shortcut = None
            
    def forward(self, x):
        out = self.main(x)
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return out


class TinierHAR(nn.Module):
    """
    TinierHAR: Ultra-lightweight HAR baseline.
    
    Architecture (from TinierHAR repo - zhaxidele/TinierHAR):
    - DepthwiseSeparableConv blocks with residual shortcuts
    - First 2 blocks with MaxPool, additional blocks without
    - Bidirectional GRU for temporal modeling  
    - Temporal attention aggregation
    - Lightweight classifier
    
    Paper: "TinierHAR: Towards Ultra-Lightweight Deep Learning Models 
           for Efficient HAR on Edge Devices" (UbiComp/ISWC 2025)
    
    Target: ~17k parameters
    """
    
    def __init__(
        self,
        numClasses: int = 6,
        inChannels: int = 9,
        seqLen: int = 128,
        nbFilters: int = 8,        # nb_filters in original
        nbConvBlocks: int = 4,     # nb_conv_blocks (2 with pool, rest without)
        gruUnits: int = 16,        # Verified: 16 gives ~17k params
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.numClasses = numClasses
        self.inChannels = inChannels
        self.seqLen = seqLen
        
        # Build conv blocks as in TinierHAR repo
        convBlocks = []
        
        # Block 1: 1 -> nbFilters, with maxpool
        convBlocks.append(TinierHARConvBlock(
            1, nbFilters, kernelSize=5, useMaxpool=True, useShortcut=True
        ))
        
        # Block 2: nbFilters -> 2*nbFilters, with maxpool
        convBlocks.append(TinierHARConvBlock(
            nbFilters, 2 * nbFilters, kernelSize=5, useMaxpool=True, useShortcut=True
        ))
        
        # Additional blocks: 2*nbFilters -> 2*nbFilters, no maxpool
        for _ in range(nbConvBlocks - 2):
            convBlocks.append(TinierHARConvBlock(
                2 * nbFilters, 2 * nbFilters, kernelSize=5, useMaxpool=False, useShortcut=True
            ))
        
        self.convBlocks = nn.Sequential(*convBlocks)
        
        # Calculate GRU input dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, seqLen, inChannels)
            out = self.convBlocks(dummy)
            # out shape: [B, C, T', inChannels]
            self.gruInputDim = out.size(1) * out.size(3)  # C * inChannels
            self.temporalLen = out.size(2)
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=self.gruInputDim,
            hidden_size=gruUnits,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Temporal attention (matching TinierHAR repo)
        self.attention = nn.Linear(gruUnits * 2, 1, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classifier
        self.classifier = nn.Linear(gruUnits * 2, numClasses)
        
        self._initWeights()
        
    def _initWeights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, T, C] or [B, C, T]
            
        Returns:
            Logits [B, numClasses]
        """
        # Input: [B, T, C] -> reshape to [B, 1, T, C] for 2D conv
        if x.dim() == 3:
            if x.shape[-1] == self.inChannels:
                # [B, T, C] format
                x = x.unsqueeze(1)  # [B, 1, T, C]
            else:
                # [B, C, T] format
                x = x.permute(0, 2, 1).unsqueeze(1)  # [B, 1, T, C]
        
        # Conv blocks: [B, 1, T, C] -> [B, C', T', C]
        x = self.convBlocks(x)
        
        # Reshape for GRU: [B, C', T', C] -> [B, T', C'*C]
        B, C, T, Cin = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, -1)
        
        # Apply dropout
        x = self.dropout(x)
        
        # GRU: [B, T', gruInputDim] -> [B, T', gruUnits*2]
        x, _ = self.gru(x)
        
        # Temporal attention aggregation
        attnWeights = torch.softmax(self.attention(x), dim=1)  # [B, T', 1]
        x = torch.sum(attnWeights * x, dim=1)  # [B, gruUnits*2]
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def countParameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {
            'total': total,
            'convBlocks': sum(p.numel() for p in self.convBlocks.parameters()),
            'gru': sum(p.numel() for p in self.gru.parameters()),
            'attention': sum(p.numel() for p in self.attention.parameters()),
            'classifier': sum(p.numel() for p in self.classifier.parameters()),
        }


def createTinierHar(dataset: str = 'ucihar') -> TinierHAR:
    """
    Factory function for dataset-specific TinierHAR models.
    
    Args:
        dataset: Dataset name
        
    Returns:
        Configured TinierHAR model
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
    return TinierHAR(**cfg)


if __name__ == '__main__':
    # Quick test
    print("=== TinierHAR (~17K) ===")
    model = TinierHAR(numClasses=6, inChannels=9, seqLen=128)
    params = model.countParameters()
    
    for k, v in params.items():
        print(f"  {k}: {v:,}")
        
    # Test forward pass
    x = torch.randn(2, 128, 9)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
