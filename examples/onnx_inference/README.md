# Onnx Inference

This folder contains examples explaining how to create [ONNX](https://github.com/onnx/onnx) models from `SentenceTransformers` for deployment.


Currently, there exists only a single [example notebook](onnx_inference.ipynb), which explains how to turn the `bert-base-nli-stsb-mean-tokens` model into an optimized ONNX model. The speedup of the optimized model can be dependent upon the hardware you use. On a V100, this speeds the model up by a factor of 3-8.