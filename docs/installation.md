# Installation

We recommend **Python 3.6** or higher, **[PyTorch 1.6.0](https://pytorch.org/get-started/locally/)** or higher and **[transformers v4.6.0](https://github.com/huggingface/transformers)** or higher. The code does **not** work with Python 2.7. 




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

**Install from sources**

Alternatively, you can also clone the latest version from the [repository](https://github.com/UKPLab/sentence-transformers) and install it directly from the source code:
````
pip install -e .
```` 

## Install PyTorch with CUDA-Support

If you want to use a GPU / CUDA, you must install PyTorch with the matching CUDA Version. Follow
[PyTorch - Get Started](https://pytorch.org/get-started/locally/) for further details how to install PyTorch.
