#! /usr/bin/env python

DESCRIPTION = "intro_to_pytorch: set of notebooks for beginners in PyTorch"

DISTNAME = 'intro_to_pytorch'
MAINTAINER = 'emlozin'
AUTHOR = 'emlozin, CaveSven, mabu-github'
URL = 'https://github.com/emlozin/intro_to_pytorch'
LICENSE = 'MIT License'
VERSION = '0.1.0'
PYTHON_REQUIRES = ">=3.7.9"

INSTALL_REQUIRES = [
    'ipykernel',
    'ipython',
    'ipywidgets',
    'ipympl',
    'jupyter',
    'jupyterlab',
    'nodejs',
    'matplotlib',
    'notebook',
    'numpy',
    'pandas',
    'pickleshare',
    'pillow',
    'torch',
    'torchvision',
    'seaborn',
    'jupyter_contrib_nbextensions',
    'jupyter_nbextensions_configurator',
]

PACKAGES = [
    'intro_to_pytorch'
]

if __name__ == "__main__":

    from setuptools import setup

    import sys
    if sys.version_info[:3] < (3, 7, 9):
        raise RuntimeError("intro_to_pytorch requires python >= 3.7.9")

    setup(
        name=DISTNAME,
        author=MAINTAINER,
        maintainer=MAINTAINER,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        packages=PACKAGES,
    )