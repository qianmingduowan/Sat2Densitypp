# python3.8
"""Collects all models."""

from .iresnet import iresnet100
from .stylegan2_generator import StyleGAN2Generator
from .s2d_eg3d_generator_v2 import EG3DGenerator as S2D_EG3DGenerator2

__all__ = ['build_model']

_MODELS = {
    'IResNet100': iresnet100,
    'StyleGAN2Generator': StyleGAN2Generator,
    'S2D_EG3DGenerator2': S2D_EG3DGenerator2,
}


def build_model(model_type, **kwargs):
    """Builds a model based on its class type.

    Args:
        model_type: Class type to which the model belongs, which is case
            sensitive.
        **kwargs: Additional arguments to build the model.

    Raises:
        ValueError: If the `model_type` is not supported.
    """
    if model_type not in _MODELS:
        raise ValueError(f'Invalid model type: `{model_type}`!\n'
                         f'Types allowed: {list(_MODELS)}.')
    return _MODELS[model_type](**kwargs)
