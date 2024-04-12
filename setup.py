"""
setup.py

Description: Used to set up PIP package
"""

# Standard libraries
from setuptools import setup, find_packages

# Set up package
setup(
    name="zero_init",
    version="0.2",
    url="https://github.com/stan-hua/ZerO-Initialization-PyTorch",
    author="Stanley Hua",
    description="PyTorch implementation of ZerO Initialization",
    packages=find_packages(),
    install_requires=["scipy>=0.8.0", "torch"],
    keywords="pytorch implementation",
    project_urls={
        "Source": "https://github.com/stan-hua/ZerO-Initialization-PyTorch",
    }
)