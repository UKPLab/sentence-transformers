# Installation

We recommend **Python 3.8+**, **[PyTorch 1.11.0+](https://pytorch.org/get-started/locally/)**, and **[transformers v4.34.0+](https://github.com/huggingface/transformers)**. There are three options to install Sentence Transformers:
* **Default:** This allows for loading, saving, and inference (i.e., getting embeddings) of models.
* **Default and Training**: All of the above plus training.
* **Development**: All of the above plus some dependencies for developing Sentence Transformers, see [Editable Install](#editable-install).

## Install with pip

```eval_rst

.. tab:: Default

    ::

        pip install -U sentence-transformers

.. tab:: Default and Training

    ::

        pip install -U "sentence-transformers[train]"

    To use `Weights and Biases <https://wandb.ai/>`_ to track your training logs, you should also install ``wandb`` **(recommended)**::

        pip install wandb
    
    And to track your Carbon Emissions while training and have this information automatically included in your model cards, also install ``codecarbon`` **(recommended)**::

        pip install codecarbon

.. tab:: Development

    ::

        pip install -U "sentence-transformers[dev]"

```

## Install with Conda

```eval_rst

.. tab:: Default

    ::

        conda install -c conda-forge sentence-transformers

.. tab:: Default and Training

    ::

        conda install -c conda-forge sentence-transformers accelerate datasets

    To use `Weights and Biases <https://wandb.ai/>`_ to track your training logs, you should also install ``wandb`` **(recommended)**::

        pip install wandb
    
    And to track your Carbon Emissions while training and have this information automatically included in your model cards, also install ``codecarbon`` **(recommended)**::

        pip install codecarbon

.. tab:: Development

    ::

        conda install -c conda-forge sentence-transformers accelerate datasets pre-commit pytest ruff

```

## Install from Source

You can install ``sentence-transformers`` directly from source to take advantage of the bleeding edge `master` branch rather than the latest stable release:

```eval_rst

.. tab:: Default

    ::

        pip install git+https://github.com/UKPLab/sentence-transformers.git

.. tab:: Default and Training

    ::

        pip install -U "sentence-transformers[train] @ git+https://github.com/UKPLab/sentence-transformers.git"

    To use `Weights and Biases <https://wandb.ai/>`_ to track your training logs, you should also install ``wandb`` **(recommended)**::

        pip install wandb
    
    And to track your carbon emissions while training and have this information automatically included in your model cards, also install ``codecarbon`` **(recommended)**::

        pip install codecarbon

.. tab:: Development

    ::

        pip install -U "sentence-transformers[dev] @ git+https://github.com/UKPLab/sentence-transformers.git"

```

## Editable Install

If you want to make changes to ``sentence-transformers``, you will need an editable install. Clone the repository and install it with these commands:
```
git clone https://github.com/UKPLab/sentence-transformers
cd sentence-transformers
pip install -e ".[train,dev]"
```

These commands will link the new `sentence-transformers` folder and your Python library paths, such that this folder will be used when importing `sentence-transformers`.

## Install PyTorch with CUDA support

To use a GPU/CUDA, you must install PyTorch with CUDA support. Follow [PyTorch - Get Started](https://pytorch.org/get-started/locally/) for installation steps.