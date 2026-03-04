"""
LightDeepConvLSTM Models Package

This package contains the LightDeepConvLSTM architecture for Human Activity Recognition.
"""

from .light_deep_conv_lstm import (
    LightDeepConvLSTM,
    createLightDeepConvLSTM,
    LIGHTDEEPCONVLSTM_CONFIG,
)

from .light_deep_conv_lstm_variants import (
    LightDeepConvLSTMVariant,
    LightDeepConvLSTMVariantSpec,
    create_variant_model,
    make_variant_spec,
)

__all__ = [
    # Main Model
    'LightDeepConvLSTM',
    'createLightDeepConvLSTM',
    'LIGHTDEEPCONVLSTM_CONFIG',

    # Ablation Variants (kept separate from frozen reference model)
    'LightDeepConvLSTMVariant',
    'LightDeepConvLSTMVariantSpec',
    'create_variant_model',
    'make_variant_spec',
]
