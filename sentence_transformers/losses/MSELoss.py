from torch import nn, Tensor
from typing import Iterable, Dict


class MSELoss(nn.Module):
    def __init__(self, model):
        """
        Computes the MSE loss between the computed sentence embedding and a target sentence embedding. This loss
        is used when extending sentence embeddings to new languages as described in our publication
        Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation.

        For an example, see `the distillation documentation <../../examples/training/distillation/README.html>`_ on extending language models to new languages.

        :param model: SentenceTransformerModel

        References:
            - Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation: https://arxiv.org/abs/2004.09813
            - `Training > Model Distillation <../../examples/training/distillation/README.html>`_
            - `Training > Multilingual Models <../../examples/training/multilingual/README.html>`_

        Requirements:
            1. Usually uses a finetuned teacher M in a knowledge distillation setup

        Relations:
            - :class:`MarginMSELoss` is equivalent to this loss, but with a margin through a negative pair.

        Input:
            +-------------------+-----------------------------+
            | Texts             | Labels                      |
            +===================+=============================+
            | single sentences  | model sentence embeddings   |
            +-------------------+-----------------------------+

        Example::

            from sentence_transformers import SentenceTransformer, InputExample, losses
            from torch.utils.data import DataLoader

            model_en = SentenceTransformer('bert-base-cased')
            model_fr = SentenceTransformer('flaubert/flaubert_base_cased')

            examples_en = ['The first sentence',  'The second sentence', 'The third sentence',  'The fourth sentence']
            examples_fr = ['La première phrase',  'La deuxième phrase', 'La troisième phrase',  'La quatrième phrase']
            train_batch_size = 2

            labels_en_en = model_en.encode(examples_en)
            examples_en_fr = [InputExample(texts=[x], label=labels_en_en[i]) for i, x in enumerate(examples_en)]
            loader_en_fr = DataLoader(examples_en_fr, batch_size=train_batch_size)

            examples_fr_fr = [InputExample(texts=[x], label=labels_en_en[i]) for i, x in enumerate(examples_fr)]
            loader_fr_fr = DataLoader(examples_fr_fr, batch_size=train_batch_size)

            train_loss = losses.MSELoss(model=model_fr)
            model_fr.fit(
                [(loader_en_fr, train_loss), (loader_fr_fr, train_loss)],
                epochs=10,
            )
        """
        super(MSELoss, self).__init__()
        self.model = model
        self.loss_fct = nn.MSELoss()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        rep = self.model(sentence_features[0])["sentence_embedding"]
        return self.loss_fct(rep, labels)
