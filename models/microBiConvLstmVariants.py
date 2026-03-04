"""
MicroBiConvLSTM ablation variants.

Provides controlled architectural ablations without modifying the frozen
reference model in microBiConvLstm.py.

Supported variants:
    A0: Base model (frozen architecture).
    A1: No MaxPool (remove pool1 and pool2).
    A2: Unidirectional LSTM (bidirectional=False).
    A3: Single Conv (remove Stage II).
    A4: Mean pooling (aggregation='mean').

Author: Mridankan Mandal
Paper: https://arxiv.org/abs/2602.06523
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .microBiConvLstm import MICRO_BI_CONV_LSTM_CONFIG


Aggregation = Literal["last", "mean"]


@dataclass(frozen=True)
class MicroBiConvLSTMVariantSpec:
    """Architecture specification for controlled ablation studies."""

    numClasses: int
    inChannels: int
    seqLen: int
    dropout: float

    convFilters: int = MICRO_BI_CONV_LSTM_CONFIG["convFilters"]
    convKernel: int = MICRO_BI_CONV_LSTM_CONFIG["convKernel"]
    convPadding: int = MICRO_BI_CONV_LSTM_CONFIG["convPadding"]
    lstmHidden: int = MICRO_BI_CONV_LSTM_CONFIG["lstmHidden"]
    lstmLayers: int = MICRO_BI_CONV_LSTM_CONFIG["lstmLayers"]
    bidirectional: bool = MICRO_BI_CONV_LSTM_CONFIG["bidirectional"]

    usePool1: bool = True
    usePool2: bool = True
    useConv2: bool = True
    aggregation: Aggregation = "last"


class MicroBiConvLSTMVariant(nn.Module):
    """Ablation-capable MicroBiConvLSTM with configurable structural changes."""

    def __init__(self, spec: MicroBiConvLSTMVariantSpec):
        super().__init__()
        self.spec = spec

        self.conv1 = nn.Conv1d(
            spec.inChannels, spec.convFilters,
            kernel_size=spec.convKernel, padding=spec.convPadding, stride=1, bias=True,
        )
        self.bn1 = nn.BatchNorm1d(spec.convFilters)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2) if spec.usePool1 else nn.Identity()

        if spec.useConv2:
            self.conv2 = nn.Conv1d(
                spec.convFilters, spec.convFilters,
                kernel_size=spec.convKernel, padding=spec.convPadding, stride=1, bias=True,
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
        """Initialize weights with best-practice defaults."""
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
                        param.data[n // 4: n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input: [B, T, C]. Output: [B, numClasses]."""
        x = x.permute(0, 2, 1)

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.pool1(x)

        if self.conv2 is not None and self.bn2 is not None:
            x = F.relu(self.bn2(self.conv2(x)), inplace=True)

        x = self.pool2(x)

        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)

        if self.spec.aggregation == "last":
            x = x[:, -1, :]
        else:
            x = x.mean(dim=1)

        x = self.dropoutLayer(x)
        return self.classifier(x)


def makeVariantSpec(
    *,
    variantId: Literal["A0", "A1", "A2", "A3", "A4"],
    numClasses: int,
    inChannels: int,
    seqLen: int,
    dropout: float,
    convFilters: Optional[int] = None,
    lstmHidden: Optional[int] = None,
) -> MicroBiConvLSTMVariantSpec:
    """Create a spec for a named ablation variant."""
    convFilters = int(convFilters) if convFilters is not None else MICRO_BI_CONV_LSTM_CONFIG["convFilters"]
    lstmHidden = int(lstmHidden) if lstmHidden is not None else MICRO_BI_CONV_LSTM_CONFIG["lstmHidden"]

    baseKwargs = dict(
        numClasses=numClasses, inChannels=inChannels, seqLen=seqLen,
        dropout=dropout, convFilters=convFilters, lstmHidden=lstmHidden,
    )

    if variantId == "A0":
        return MicroBiConvLSTMVariantSpec(**baseKwargs, bidirectional=True, usePool1=True, usePool2=True, useConv2=True, aggregation="last")
    if variantId == "A1":
        return MicroBiConvLSTMVariantSpec(**baseKwargs, bidirectional=True, usePool1=False, usePool2=False, useConv2=True, aggregation="last")
    if variantId == "A2":
        return MicroBiConvLSTMVariantSpec(**baseKwargs, bidirectional=False, usePool1=True, usePool2=True, useConv2=True, aggregation="last")
    if variantId == "A3":
        return MicroBiConvLSTMVariantSpec(**baseKwargs, bidirectional=True, usePool1=True, usePool2=False, useConv2=False, aggregation="last")
    if variantId == "A4":
        return MicroBiConvLSTMVariantSpec(**baseKwargs, bidirectional=True, usePool1=True, usePool2=True, useConv2=True, aggregation="mean")

    raise ValueError(f"Unknown variantId: {variantId}")


def createVariantModel(spec: MicroBiConvLSTMVariantSpec) -> MicroBiConvLSTMVariant:
    """Create a variant model from a spec."""
    return MicroBiConvLSTMVariant(spec)
