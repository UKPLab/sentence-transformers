import json
from enum import IntEnum


class LossFunction(IntEnum):
    """
    The loss function for the training of the model
    """
    SOFTMAX = 0
    COSINE_SIMILARITY = 1
    TRIPLET_LOSS = 2
    BATCH_HARD_TRIPLET_LOSS = 3
    MULTIPLE_NEGATIVES_RANKING_LOSS = 4


class TripletMetric(IntEnum):
    """
    The metric for the triplet loss
    """
    COSINE = 0
    EUCLIDEAN = 1
    MANHATTAN = 2


class SentenceTransformerConfig:
    """
    Configuration of a Sentence Transformer model
    """
    model: str
    tokenizer_model: str
    do_lower_case: bool
    max_seq_length: int
    pooling_mode_cls_token: bool
    pooling_mode_mean_tokens: bool
    pooling_mode_max_tokens: bool
    loss_function: LossFunction

    softmax_num_labels: int
    softmax_concatenation_sent_rep: bool
    softmax_concatenation_sent_difference: bool
    softmax_concatenation_sent_multiplication: bool

    triplet_margin: float
    triplet_metric: TripletMetric

    def __init__(self,
                 model = "sentence_transformers.models.BERT",
                 tokenizer_model: str = "bert-base-uncased",
                 do_lower_case: bool = True,
                 max_seq_length: int = 64,
                 pooling_mode_cls_token: bool = False,
                 pooling_mode_mean_tokens: bool = True,
                 pooling_mode_max_tokens: bool = False,
                 loss_function: LossFunction = LossFunction.SOFTMAX,
                 softmax_num_labels: int = 1,
                 softmax_concatenation_sent_rep: bool = True,
                 softmax_concatenation_sent_difference: bool = True,
                 softmax_concatenation_sent_multiplication: bool = True,
                 triplet_margin: float = 1.0,
                 triplet_metric: TripletMetric = TripletMetric.COSINE):
        """
        The configuration for a Sentence BERT model

        :param tokenizer_model:
            the underlying BERT model from pytorch_pretrained_bert
        :param do_lower_case:
            convert sentences to lower case during encoding and embedding
        :param max_seq_length:
            the maximal sequence length of a tokenized sentence. Tokens after the maximal length are cut off
        :param pooling_mode_cls_token:
            Concatenate the first token of the transformer (CLS token) to the embedding
        :param pooling_mode_mean_tokens:
            Compute element-wise mean of all tokens and concatenate it to the embedding
        :param pooling_mode_max_tokens:
            Compute element-wise max of all tokens and concatenate it to the embedding
        :param loss_function:
            the loss function used during training
            SOFTMAX: classification with labeled data
            COSINE_SIMILARITY: minimize cosine between embeddings. No label needed for training
            TRIPLET: Minimize distance to positive example and maximize distance to negative example for an anchor.
                The order *must* be anchor, positive, negative for the InputExamples.
            BATCH_HARD_TRIPLET_LOSS: Triplet loss using the hardest positive and hardest negative
                of a batch to form a triplet.
                See sbert.losses.batch_hard_triplet_loss for more information.
            MULTIPLE_NEGATIVES_RANKING_LOSS: Ranking loss using the batch for the ranking.
                See sbert.losses.multiple_negatives_ranking_loss for more information.
        :param softmax_num_labels:
            the number of class labels during training with SBERTLossFunction.SOFTMAX
        :param softmax_concatenation_sent_rep:
            concatenate the sentence embeddings to the input for the classification layer during training
            with SBERTLossFunction.SOFTMAX
        :param softmax_concatenation_sent_difference:
            concatenate the difference between the sentence embeddings to the input for the classification
            layer during training with SBERTLossFunction.SOFTMAX
        :param softmax_concatenation_sent_multiplication:
            concatenate the elementwise product of the sentence embeddings to the input for the classification
            layer during training with SBERTLossFunction.SOFTMAX
        :param triplet_margin
            the margin for the triplet loss.
            The loss is max(0, margin + distance(anchor, positive) - distance(anchor, negative))
        :param triplet_metric
            the metric used for the distance
        """
        if isinstance(model, str):
            self.model = model
        else:
            self.model = model.__module__

        self.triplet_metric = triplet_metric
        self.triplet_margin = triplet_margin
        self.tokenizer_model = tokenizer_model
        self.do_lower_case = do_lower_case
        self.max_seq_length = max_seq_length
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.loss_function = loss_function
        self.softmax_num_labels = softmax_num_labels
        self.softmax_concatenation_sent_difference = softmax_concatenation_sent_difference
        self.softmax_concatenation_sent_multiplication = softmax_concatenation_sent_multiplication
        self.softmax_concatenation_sent_rep = softmax_concatenation_sent_rep

    def to_json_file(self, path: str):
        """
        Serialize the config to a JSON file at the given path
        :param path:
            the path for the JSON file
        """
        with open(path, "w", encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=2)

    def __repr__(self):
        return str(json.dumps(self.__dict__, indent=2) + "\n")

    @staticmethod
    def from_json_file(path: str):  # -> SBERTConfig : unresolved reference problem
        """
        Load a config from the given JSON
        :param path:
            the path to the JSON
        :return: the SBERTConfig created from the JSON file
        """
        json_config = json.load(open(path, "r", encoding="utf-8"))
        config = SentenceTransformerConfig()
        for key, value in json_config.items():
            config.__dict__[key] = value
        return config
