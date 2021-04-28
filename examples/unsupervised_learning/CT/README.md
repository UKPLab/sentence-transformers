# CT

This folder shows an example, how can we train an unsupervised [CT (Contrastive Tension)](https://openreview.net/pdf?id=Ov_sMNau-PF) model with pure sentences as training data.

## Background
During training, CT builds two independent encoders ('Model1' and 'Model2') with intial parameters shared to encode a pair of sentences Each training batch contains multiple mini-batches. For the example of K=7,  each mini-batch consists of sentence pairs (S_A, S_A), (S_A, S_B), (S_A, S_C), ..., (S_A, S_H) and the corresponding labels are 1, 0, 0, ..., 0. In other words, one identical pair of sentences is viewed as the positive example and other pairs of different sentences are viewed as the negative examples (i.e. 1 positive + K negative pairs). The training objective is the binary cross-entropy between the generated similarity scores and labels. This example is illustrated in the figure (from the Appendix A.1 of the CT paper) below:![](CT.jpg)

After training, the model 2 will be used for inference, which usually has better performance.

## Training and evaluation on STSb data

One can simply run this command to train and evaluate a CT model on STSb with the labels removed:

```python train_STSb.py```

This can achieve around 0.75 Spearman's rank correlation score with only 11K unlabeled sentences.