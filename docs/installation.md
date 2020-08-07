# Installation

We recommend **Python 3.6** or higher, **[PyTorch 1.2.0](https://pytorch.org/get-started/locally/)** or higher and **[transformers v3.0.2](https://github.com/huggingface/transformers)** or higher. The code does **not** work with Python 2.7.


## Install PyTorch
First, follow the installation for PyTroch you can find here: [PyTorch - Get Started](https://pytorch.org/get-started/locally/). As the provided models can have a high computational overhead, it is recommend to run them on a GPU. See the PyTorch page how to install PyTorch for GPU (CUDA).


## Install SentenceTransformers

**Install with pip**

Install the *sentence-transformers* with `pip`:
```
pip install -U sentence-transformers
```

**Install from sources**

Alternatively, you can also clone the latest version from the [repository](https://github.com/UKPLab/sentence-transformers) and install it directly from the source code:
````
pip install -e .
```` 