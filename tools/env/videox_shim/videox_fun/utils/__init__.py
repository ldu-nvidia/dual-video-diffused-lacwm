"""Minimal VideoX-Fun utility export required by WanTransformer3DModel."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from .cfg_optimization import cfg_skip

__all__ = ["cfg_skip"]
