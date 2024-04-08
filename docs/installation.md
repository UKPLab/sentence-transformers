# Installation

We recommend **Python 3.8** or higher, **[PyTorch 1.11.0](https://pytorch.org/get-started/locally/)** or higher and **[transformers v4.32.0](https://github.com/huggingface/transformers)** or higher.

## Install SentenceTransformers

**Install with pip**

Install the *sentence-transformers* with `pip`:
```
pip install -U sentence-transformers
```

**Install with conda**

Apple silicon Installation of *sentence-transformers*
```
conda install -c conda-forge sentence-transformers
```

**Install from source**

Alternatively, you can also clone the latest version from the [repository](https://github.com/UKPLab/sentence-transformers) and install it directly from the source code:
````
pip install -e .
```` 

## Install PyTorch with CUDA support

If you want to use a GPU / CUDA, you must install PyTorch with the matching CUDA Version. Follow
[PyTorch - Get Started](https://pytorch.org/get-started/locally/) for further details how to install PyTorch.
