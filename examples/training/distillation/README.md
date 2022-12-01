# Model Distillation 
This folder contains example to make SentenceTransformer models **faster, cheaper and lighter**. These light models achieve 97.5% - 100% performance of the original model on downstream tasks.

## Knowledge Distillation
See: **[model_distillation.py](model_distillation.py)**

Knowledge distillation describes the process to transfer knowledge from a  teacher model to a student model. It can be used to extend sentence embeddings to new languages ([Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813)), but the traditional approach is to have slow (but well performing) teacher model and a fast student model.

The fast student model imitates the teacher model and achieves by this a high performance. 

![Knowledge Distillation](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/monolingual-distillation.png)


**[model_distillation.py](model_distillation.py)** implements two options for creating the student model:
1) Use a light transformer model like TinyBERT or BERT-Small to imitate the teacher.
2) We take the teacher model and keep only certain layers, for example, only 4 layers.

Option 2) works usually better, as we keep most of the weights from the teacher. In Option 1, we have to tune all
weights in the student from scratch.

## Speed - Performance Trade-Off
Smaller models are faster, but show a (slightly) worse performance when evaluated on down stream tasks. To get an impression of this trade-off, we show some numbers of the *stsb-roberta-base* model with different number of layers:

| Layers | STSbenchmark Performance | Performance Decrease |Speed (Sent. / Sec. on V100-GPU) |
| ---- |:----:|:----:|:----:|
| teacher: 12 | 85.44 | - | 2300 |
| 8 | 85.54 | +0.1% | 3200 |
| 6 | 85.23 | -0.2% | 4000 |
| 4 | 84.92 | -0.6% | 5300 |
| 3 |  84.39 | -1.2%  |6500 |
| 2 | 83.32 | -2.5% | 7700 |
| 1 | 80.86 |  -5.4%| 9200 |


## Dimensionality Reduction
By default, the pretrained models output embeddings with size 768 (base-models) or with size 1024 (large-models). However, when you store Millions of embeddings, this can require quite a lot of memory / storage.

**[dimensionality_reduction.py](dimensionality_reduction.py)** contains a simple example how to reduce the embedding dimension to any size by using Principle Component Analysis (PCA). In that example, we reduce 768 dimension to 128 dimension, reducing the storage requirement by factor 6. The performance only slightly drops from 85.44 to 84.96 on the STS benchmark dataset.

This dimensionality reduction technique can easily be applied to existent models. We could even reduce the embeddings size to 32, reducing the storage requirement by factor 24 (performance decreases to 81.82). 

Note: This technique neither improves the runtime, nor the memory requirement for running the model. It only reduces the needed space to store embeddings, for example, for [semantic search](../../applications/semantic-search/README.md).

## Quantization
A [quantized model](https://pytorch.org/docs/stable/quantization.html) executes some or all of the operations with integers rather than floating point values. This allows for a more compact models and the use of high performance vectorized operations on many hardware platforms.

For models that are run on **CPUs**, this can yield 40% smaller models and a faster inference time: Depending on the CPU, speedup are between 15% and 400%. Model quantization is (as of now) not supported for GPUs by PyTorch.

For an example, see [model_quantization.py](model_quantization.py)
