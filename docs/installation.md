# Installation

We recommend **Python 3.8+**, **[PyTorch 1.11.0+](https://pytorch.org/get-started/locally/)**, and **[transformers v4.34.0+](https://github.com/huggingface/transformers)**.

## Install with pip

Install the *sentence-transformers* with `pip`:
```
pip install -U sentence-transformers
```

## Install with conda

Apple silicon installation of *sentence-transformers*
```
conda install -c conda-forge sentence-transformers
```

## Install from source

You can install *sentence-transformers* directly from source to take advantage of the bleeding edge `master` branch rather than the latest stable release:
```
pip install git+https://github.com/UKPLab/sentence-transformers
```

## Editable install

If you want to make changes to *sentence-transformers*, you will need an editable install. Clone the repository and install it with these commands:
```
git clone https://github.com/UKPLab/sentence-transformers
cd sentence-transformers
pip install -e .
```

These commands will link the new `sentence-transformers` folder and your Python library paths, such that this folder will be used when importing `sentence-transformers`.

## Install PyTorch with CUDA support

To use a GPU/CUDA, you must install PyTorch with CUDA support. Follow [PyTorch - Get Started](https://pytorch.org/get-started/locally/) for installation steps.