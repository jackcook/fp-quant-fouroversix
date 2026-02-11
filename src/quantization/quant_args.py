from enum import Enum


class QuantizationFormat(str, Enum):
    """
    Enum storing quantization format options
    """
    INT = "int"
    FP = "fp"
    NVFP = "nvfp"
    MXFP = "mxfp"

class QuantizationGranularity(str, Enum):
    """
    Enum storing quantization granularity options
    """
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"

class QuantizationObserver(str, Enum):
    """
    Enum storing quantization observer options
    """
    MINMAX = "minmax"
    MSE = "mse"

class QuantizationOrder(str, Enum):
    """
    Enum storing quantization order options
    """
    DEFAULT = "default"
    ACTIVATION = "activation"

class ScalePrecision(str, Enum):
    """
    Enum scale precision options
    """
    FP16 = "fp16"
    E4M3 = "e4m3"
    E8M0 = "e8m0"

class ScaleSelectionRule(str, Enum):
    """
    Enum scale selection rule options
    """
    STATIC_6 = "static_6"
    STATIC_4 = "static_4"
    MAE = "mae"
    MSE = "mse"
    ABS_MAX = "abs_max"