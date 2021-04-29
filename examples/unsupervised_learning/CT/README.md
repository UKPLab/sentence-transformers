# CT
Carlsson et al. present in [Semantic Re-Tuning With Contrastive Tension (CT)](https://openreview.net/pdf?id=Ov_sMNau-PF) an unsupervised learning approach for sentence embeddings that just requires sentences.

## Background
During training, CT builds two independent encoders ('Model1' and 'Model2') with intial parameters shared to encode a pair of sentences Each training batch contains multiple mini-batches. For the example of K=7,  each mini-batch consists of sentence pairs (S_A, S_A), (S_A, S_B), (S_A, S_C), ..., (S_A, S_H) and the corresponding labels are 1, 0, 0, ..., 0. In other words, one identical pair of sentences is viewed as the positive example and other pairs of different sentences are viewed as the negative examples (i.e. 1 positive + K negative pairs). The training objective is the binary cross-entropy between the generated similarity scores and labels. This example is illustrated in the figure (from the Appendix A.1 of the CT paper) below:

![CT working](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/CT.jpg)

After training, the model 2 will be used for inference, which usually has better performance.

## Training Examples 

- **[train_stsb_ct.py](train_stsb_ct.py)**: This example uses 1 million sentences from Wikipedia to train with CT. It evaluate the performance on the  [STSbenchmark dataset](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark).
- **[train_askubuntu_ct.py](train_askubuntu_ct.py)**: This example trains on [AskUbuntu Questions dataset](https://github.com/taolei87/askubuntu), a dataset with questions from the AskUbuntu Stackexchange forum.