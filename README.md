# Introduction to PyTorch

This repository contains a set of notebooks for beginners in PyTorch.

### Requirements

For specific requirements refer to ``requirements.txt`` file. 

### Installation

#### Binder

You can use Binder. It does not require any installation, just click the button below:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/emlozin/intro_to_pytorch/main)

#### Conda
Make sure you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed.

Linux or Mac
```
conda env create -f environment.yml
conda activate intro_to_pytorch
python -m ipykernel install --user --name=intro_to_pytorch
jupyter notebook
```

Windows
```
conda env create -f environment.yml
conda activate intro_to_pytorch
conda install pywin32
python -m ipykernel install --user --name=intro_to_pytorch
jupyter notebook
```

## References

During development of these notebooks, following sources have been used as a source of inspiration:

- [Deep Learning (PyTorch) by Udacity](https://github.com/udacity/deep-learning-v2-pytorch)
