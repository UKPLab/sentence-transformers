from enum import Enum
from typing import Iterable, Dict
import torch.nn.functional as F
from torch import nn, Tensor
from sentence_transformers.SentenceTransformer import SentenceTransformer


class SiameseDistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the
    two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.

    Further information: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    :param model: SentenceTransformer model
    :param distance_metric: Function that returns a distance between two embeddings. The class SiameseDistanceMetric contains pre-defined metrices that can be used
    :param margin: Negative samples (label == 0) should have a distance of at least the margin value.
    :param size_average: Average by the size of the mini-batch.

    Example::

        from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
        from torch.utils.data import DataLoader

        model = SentenceTransformer('all-MiniLM-L6-v2')
        train_examples = [
            InputExample(texts=['This is a positive pair', 'Where the distance will be minimized'], label=1),
            InputExample(texts=['This is a negative pair', 'Their distance will be increased'], label=0)]

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
        train_loss = losses.ContrastiveLoss(model=model)

        model.fit([(train_dataloader, train_loss)], show_progress_bar=True)

    """

    def __init__(
        self,
        model: SentenceTransformer,
        distance_metric=SiameseDistanceMetric.COSINE_DISTANCE,
        margin: float = 0.5,
        size_average: bool = True,
    ):
        super(ContrastiveLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin
        self.model = model
        self.size_average = size_average

    def get_config_dict(self):
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(SiameseDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = "SiameseDistanceMetric.{}".format(name)
                break

        return {"distance_metric": distance_metric_name, "margin": self.margin, "size_average": self.size_average}

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        assert len(reps) == 2
        rep_anchor, rep_other = reps
        distances = self.distance_metric(rep_anchor, rep_other)
        losses = 0.5 * (
            labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2)
        )
        return losses.mean() if self.size_average else losses.sum()
