import torch

import sevenn._keys as KEY

from .convolution import IrrepsConvolution, IrrepsScatterGatterFusedConvolution

try:
    from .swift_backend import SwiftTPConvolution

    _SWIFT_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - import is environment dependent
    SwiftTPConvolution = None  # type: ignore[assignment]
    _SWIFT_IMPORT_ERROR = exc


def is_swift_available() -> bool:
    return _SWIFT_IMPORT_ERROR is None and torch.cuda.is_available()


def swift_needed(func):
    def wrapper(*args, **kwargs):
        if is_swift_available():
            return func(*args, **kwargs)
        raise ImportError('SWIFT-TP is not available')

    return wrapper


@swift_needed
def patch_convolution(irreps_convolution: IrrepsConvolution):
    assert not irreps_convolution.layer_instantiated, (
        'Convolution layer already instantiated; cannot patch'
    )

    ret = IrrepsScatterGatterFusedConvolution.from_irreps_convolution(
        irreps_convolution
    )
    ret.convolution_cls = SwiftTPConvolution  # type: ignore[assignment]
    ret.key_filter = KEY.EDGE_VEC
    return ret
