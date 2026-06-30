"""Narrow namespace overlay for the pinned VideoX-Fun checkout.

The upstream package contains the implementation. This overlay only prevents
unrelated model families from being imported when lacwm requests the Wan
submodules.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
