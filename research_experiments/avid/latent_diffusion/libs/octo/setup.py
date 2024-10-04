from setuptools import find_packages, setup

setup(
    name="octo",
    packages=find_packages(),
    install_requires=[
        "tensorflow == 2.15.0",
        "tensorflow_datasets == 4.9.2",
        "absl-py >= 0.12.0",
        "tqdm >= 4.60.0",
        "dlimp @ git+https://github.com/kvablack/dlimp@d08da3852c149548aaa8551186d619d87375df08",
        "tensorflow_graphics==2021.12.3",
    ],
)
