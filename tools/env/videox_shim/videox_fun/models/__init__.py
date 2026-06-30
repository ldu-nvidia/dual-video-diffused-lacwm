"""Expose pinned VideoX-Fun model submodules without eager bulk imports."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
