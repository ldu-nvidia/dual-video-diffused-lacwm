# module_test_utils.py
import ast
import importlib
import inspect
import os


def find_python_modules(package_prefixes):
    """Find all modules in the specified packages.

    Args:
        package_prefixes (list[str] or str): Package prefix(es) to search for modules

    Returns:
        list[str]: List of fully qualified module names
    """
    if isinstance(package_prefixes, str):
        package_prefixes = [package_prefixes]

    all_modules = []

    for module_prefix in package_prefixes:
        try:
            # Import the package
            base_package = importlib.import_module(module_prefix)
            package_path = os.path.dirname(base_package.__file__)
            package_name = module_prefix.split(".")[-1]  # Extract the subpackage name

            for dirpath, _, filenames in os.walk(package_path):
                if "__pycache__" in dirpath:
                    continue

                for filename in filenames:
                    if filename.endswith(".py") and filename != "__init__.py":
                        # Compute the module path from the file path
                        rel_path = os.path.relpath(
                            os.path.join(dirpath, filename),
                            os.path.dirname(package_path),
                        )
                        module_path = os.path.splitext(rel_path)[0].replace(os.sep, ".")

                        # Extract module components and handle subpackage name correctly
                        module_components = module_path.split(".")
                        if (
                            len(module_components) > 0
                            and module_components[0] == package_name
                        ):
                            # Remove the redundant subpackage prefix if present
                            module_components = module_components[1:]

                        module_path = ".".join(module_components)
                        module_name = (
                            f"{module_prefix}.{module_path}"
                            if module_path
                            else module_prefix
                        )
                        all_modules.append(module_name)

        except ImportError as e:
            print(f"Warning: Could not import package {module_prefix}: {e}")

    return all_modules


def has_main_check(module_object):
    """Check if a module has an __main__ check.

    Args:
        module_object: The imported module object to check

    Returns:
        bool: True if the module has a __main__ check, False otherwise
    """
    try:
        # Get the source code of the module
        source = inspect.getsource(module_object)

        # Parse the source into an AST
        tree = ast.parse(source)

        # Look for __name__ == '__main__' pattern
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                if (
                    isinstance(node.left, ast.Name)
                    and node.left.id == "__name__"
                    and any(isinstance(op, ast.Eq) for op in node.ops)
                    and any(
                        isinstance(comparator, ast.Constant)
                        and comparator.value == "__main__"
                        for comparator in node.comparators
                    )
                ):
                    return True

        return False
    except Exception as e:
        # Print the exception for debugging
        print(f"Error checking for __main__ in {module_object.__name__}: {str(e)}")
        return False


def get_short_path(module_name):
    """Get a readable short path for a module name.

    Args:
        module_name (str): The full module name

    Returns:
        str: A shortened, more readable path
    """
    path_parts = module_name.split(".")
    # Skip the common prefix parts to make the output cleaner
    if len(path_parts) > 2:
        path_parts = path_parts[1:]  # Skip the first part (e.g., 'robot_wm')
    return "/".join(path_parts)
