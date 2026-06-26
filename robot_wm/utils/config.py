import importlib
import os

from hydra import compose, initialize
from hydra.utils import instantiate


def get_robot_wm_path():
    spec = importlib.util.find_spec("robot_wm")
    if spec and spec.submodule_search_locations:
        robot_wm_paths = spec.submodule_search_locations
    folder = robot_wm_paths[0]
    return os.path.dirname(folder)


def from_config(config_path, overrides=None):
    """
    Receives a config in relative robot_wm path and returns the hydra instantiated object

    Args:
        config_path: Path to the config file
        overrides: List of overrides in the format ["key1=value1", "key2=value2"]
    Returns:
        Instantiated config object
    """
    path = "/".join(config_path.split("/")[:-1])
    name = config_path.split("/")[-1].replace(".yaml", "")

    # Default to empty list if no overrides provided
    if overrides is None:
        overrides = []

    with initialize(version_base=None, config_path=f"../{path}"):
        cfg = compose(config_name=name, overrides=overrides)

    # Instantiate
    return instantiate(cfg)
