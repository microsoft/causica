from setuptools import find_packages, setup

setup(
    name="avid_utils",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["pytorch_lightning", "torch", "einops", "numpy", "cd-fvd", "pytorch_fid", "torchmetrics"],
)
