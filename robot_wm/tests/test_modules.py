# test_datasets_menagerie_modules.py
import importlib

import pytest
from testing_core import find_python_modules, get_short_path, has_main_check

# Packages to test
PACKAGES = ["robot_wm.datasets", "robot_wm.modeling", "robot_wm.utils"]


@pytest.mark.parametrize("module_name", find_python_modules(PACKAGES))
def test_module_importable(module_name):
    """Test that a module can be imported and has a __main__ check."""
    try:
        # Import the module
        module = importlib.import_module(module_name)

        # Check for __main__ check
        if not has_main_check(module):
            short_path = get_short_path(module_name)
            pytest.skip(
                f"This file is missing a __main__ and potentially untested. Verify {short_path}.py"
            )

        assert True

    except Exception as e:
        short_path = get_short_path(module_name)
        pytest.fail(f"Failed to import {short_path}: {str(e)}")
