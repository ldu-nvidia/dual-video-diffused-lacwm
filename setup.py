from setuptools import find_packages, setup

setup(
    name="robot_wm",
    version="0.2.0",
    description="LACWM with experimental video/time-frequency dual flow matching",
    packages=find_packages(include=["robot_wm", "robot_wm.*"]),
)
