"""
MicroBiConvLSTM Models Package.

Contains the MicroBiConvLSTM architecture and ablation variants for Human Activity Recognition.
"""

from .microBiConvLstm import (
    MicroBiConvLSTM,
    createMicroBiConvLstm,
    MICRO_BI_CONV_LSTM_CONFIG,
)

from .microBiConvLstmVariants import (
    MicroBiConvLSTMVariant,
    MicroBiConvLSTMVariantSpec,
    createVariantModel,
    makeVariantSpec,
)

__all__ = [
    'MicroBiConvLSTM',
    'createMicroBiConvLstm',
    'MICRO_BI_CONV_LSTM_CONFIG',
    'MicroBiConvLSTMVariant',
    'MicroBiConvLSTMVariantSpec',
    'createVariantModel',
    'makeVariantSpec',
]
