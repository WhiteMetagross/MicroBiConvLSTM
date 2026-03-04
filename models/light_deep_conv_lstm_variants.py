"""LightDeepConvLSTM ablation variants.

This module intentionally does NOT modify the frozen reference model in
`models/light_deep_conv_lstm.py`.

It provides a controlled way to instantiate architectural ablations for
systematic studies while keeping the reference implementation immutable.

Ablations supported (matching the user's study plan):
- A0: Base model (frozen architecture)
- A1: No MaxPool (remove pool1 and pool2)
- A2: Unidirectional (bidirectional=False)
- A3: Single Conv (remove Stage II: conv2/bn2/pool2)
- A4: Mean pooling (aggregation='mean')

Notes
-----
- Training hyperparameters are handled by the ablation runner.
- The forward signature matches the reference: input is [B, T, C].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .light_deep_conv_lstm import LIGHTDEEPCONVLSTM_CONFIG


Aggregation = Literal["last", "mean"]


@dataclass(frozen=True)
class LightDeepConvLSTMVariantSpec:
    """Architecture spec for controlled ablations."""

    # Dataset/model IO
    numClasses: int
    inChannels: int
    seqLen: int

    # Training-tunable (kept here so checkpoints are self-describing)
    dropout: float

    # Architecture knobs (default to frozen reference)
    convFilters: int = LIGHTDEEPCONVLSTM_CONFIG["convFilters"]
    convKernel: int = LIGHTDEEPCONVLSTM_CONFIG["convKernel"]
    convPadding: int = LIGHTDEEPCONVLSTM_CONFIG["convPadding"]
    lstmHidden: int = LIGHTDEEPCONVLSTM_CONFIG["lstmHidden"]
    lstmLayers: int = LIGHTDEEPCONVLSTM_CONFIG["lstmLayers"]
    bidirectional: bool = LIGHTDEEPCONVLSTM_CONFIG["bidirectional"]

    # Structural ablations
    usePool1: bool = True
    usePool2: bool = True
    useConv2: bool = True

    aggregation: Aggregation = "last"


class LightDeepConvLSTMVariant(nn.Module):
    """Ablation-capable LightDeepConvLSTM.

    Designed to stay close to the frozen reference while enabling
    controlled structural changes.
    """

    def __init__(self, spec: LightDeepConvLSTMVariantSpec):
        super().__init__()
        self.spec = spec

        self.conv1 = nn.Conv1d(
            spec.inChannels,
            spec.convFilters,
            kernel_size=spec.convKernel,
            padding=spec.convPadding,
            stride=1,
            bias=True,  # Match baselines/deepConvLstm.py
        )
        self.bn1 = nn.BatchNorm1d(spec.convFilters)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2) if spec.usePool1 else nn.Identity()

        if spec.useConv2:
            self.conv2 = nn.Conv1d(
                spec.convFilters,
                spec.convFilters,
                kernel_size=spec.convKernel,
                padding=spec.convPadding,
                stride=1,
                bias=True,  # Match baselines/deepConvLstm.py
            )
            self.bn2 = nn.BatchNorm1d(spec.convFilters)
        else:
            self.conv2 = None
            self.bn2 = None

        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2) if spec.usePool2 else nn.Identity()

        self.lstm = nn.LSTM(
            input_size=spec.convFilters,
            hidden_size=spec.lstmHidden,
            num_layers=spec.lstmLayers,
            batch_first=True,
            bidirectional=spec.bidirectional,
        )

        lstmOutput = spec.lstmHidden * 2 if spec.bidirectional else spec.lstmHidden
        self.dropoutLayer = nn.Dropout(spec.dropout)
        self.classifier = nn.Linear(lstmOutput, spec.numClasses)

        self._initWeights()

    def _initWeights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
                        n = param.size(0)
                        param.data[n // 4 : n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, T, C]
        x = x.permute(0, 2, 1)  # [B, C, T]

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.pool1(x)

        if self.conv2 is not None and self.bn2 is not None:
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x, inplace=True)

        x = self.pool2(x)

        x = x.permute(0, 2, 1)  # [B, T', F]
        x, _ = self.lstm(x)

        if self.spec.aggregation == "last":
            x = x[:, -1, :]
        else:
            x = x.mean(dim=1)

        x = self.dropoutLayer(x)
        return self.classifier(x)


def make_variant_spec(
    *,
    variantId: Literal["A0", "A1", "A2", "A3", "A4"],
    numClasses: int,
    inChannels: int,
    seqLen: int,
    dropout: float,
    convFilters: Optional[int] = None,
    lstmHidden: Optional[int] = None,
) -> LightDeepConvLSTMVariantSpec:
    """Create a spec for a named ablation variant."""

    convFilters = int(convFilters) if convFilters is not None else LIGHTDEEPCONVLSTM_CONFIG["convFilters"]
    lstmHidden = int(lstmHidden) if lstmHidden is not None else LIGHTDEEPCONVLSTM_CONFIG["lstmHidden"]

    if variantId == "A0":
        return LightDeepConvLSTMVariantSpec(
            numClasses=numClasses,
            inChannels=inChannels,
            seqLen=seqLen,
            dropout=dropout,
            convFilters=convFilters,
            lstmHidden=lstmHidden,
            bidirectional=True,
            usePool1=True,
            usePool2=True,
            useConv2=True,
            aggregation="last",
        )

    if variantId == "A1":
        return LightDeepConvLSTMVariantSpec(
            numClasses=numClasses,
            inChannels=inChannels,
            seqLen=seqLen,
            dropout=dropout,
            convFilters=convFilters,
            lstmHidden=lstmHidden,
            bidirectional=True,
            usePool1=False,
            usePool2=False,
            useConv2=True,
            aggregation="last",
        )

    if variantId == "A2":
        return LightDeepConvLSTMVariantSpec(
            numClasses=numClasses,
            inChannels=inChannels,
            seqLen=seqLen,
            dropout=dropout,
            convFilters=convFilters,
            lstmHidden=lstmHidden,
            bidirectional=False,
            usePool1=True,
            usePool2=True,
            useConv2=True,
            aggregation="last",
        )

    if variantId == "A3":
        return LightDeepConvLSTMVariantSpec(
            numClasses=numClasses,
            inChannels=inChannels,
            seqLen=seqLen,
            dropout=dropout,
            convFilters=convFilters,
            lstmHidden=lstmHidden,
            bidirectional=True,
            usePool1=True,
            usePool2=False,  # Stage II removed => only one pooling stage
            useConv2=False,
            aggregation="last",
        )

    if variantId == "A4":
        # Same as base model but aggregation changes; intended for eval-only.
        return LightDeepConvLSTMVariantSpec(
            numClasses=numClasses,
            inChannels=inChannels,
            seqLen=seqLen,
            dropout=dropout,
            convFilters=convFilters,
            lstmHidden=lstmHidden,
            bidirectional=True,
            usePool1=True,
            usePool2=True,
            useConv2=True,
            aggregation="mean",
        )

    raise ValueError(f"Unknown variantId: {variantId}")


def create_variant_model(spec: LightDeepConvLSTMVariantSpec) -> LightDeepConvLSTMVariant:
    return LightDeepConvLSTMVariant(spec)
