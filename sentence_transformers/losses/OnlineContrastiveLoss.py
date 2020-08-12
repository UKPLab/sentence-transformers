from typing import Iterable, Dict
import torch.nn.functional as F
from torch import nn, Tensor
from .ContrastiveLoss import SiameseDistanceMetric
from sentence_transformers.SentenceTransformer import SentenceTransformer


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss. Similar to ConstrativeLoss, but it selects hard positive (positives that are far apart)
     and hard negative pairs (negatives that are close) and computes the loss only for these pairs. Often yields
     better performances than  ConstrativeLoss.

    :param model: SentenceTransformer model
    :param distance_metric: Function that returns a distance between two emeddings. The class SiameseDistanceMetric contains pre-defined metrices that can be used
    :param margin: Negative samples (label == 0) should have a distance of at least the margin value.
    :param size_average: Average by the size of the mini-batch.

    Example::

        from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
        from sentence_transformers.readers import InputExample

        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        train_examples = [InputExample(texts=['This is a positive pair', 'Where the distance will be minimized'], label=1),
            InputExample(texts=['This is a negative pair', 'Their distance will be increased'], label=0)]
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.OnlineContrastiveLoss(model=model)
    """

    def __init__(self, model: SentenceTransformer, distance_metric=SiameseDistanceMetric.COSINE_DISTANCE, margin: float = 0.5):
        super(OnlineContrastiveLoss, self).__init__()
        self.model = model
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, size_average=False):
        embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        distance_matrix = self.distance_metric(embeddings[0], embeddings[1])
        negs = distance_matrix[labels == 0]
        poss = distance_matrix[labels == 1]

        # select hard positive and hard negative pairs
        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        positive_loss = positive_pairs.pow(2).sum()
        negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()
        loss = positive_loss + negative_loss
        return loss