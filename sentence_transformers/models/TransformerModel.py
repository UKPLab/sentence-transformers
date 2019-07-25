import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple, List
from ..losses import batch_hard_triplet_loss, multiple_negatives_ranking_loss
from .. import LossFunction
from .. import TripletMetric
from .. import SentenceTransformerConfig



class TransformerModel:
    def __init__(self, sentence_transformer_config: SentenceTransformerConfig = None):
        self.sentence_transformer_config = sentence_transformer_config
        self.classifier = None

        if sentence_transformer_config is not None:
            self.set_config(sentence_transformer_config)

    def set_config(self, sentence_transformer_config: SentenceTransformerConfig):
        self.sentence_transformer_config = sentence_transformer_config

        num_vectors_concatenated = 0
        if sentence_transformer_config.softmax_concatenation_sent_rep:
            num_vectors_concatenated += 2
        if sentence_transformer_config.softmax_concatenation_sent_difference:
            num_vectors_concatenated += 1
        if sentence_transformer_config.softmax_concatenation_sent_multiplication:
            num_vectors_concatenated += 1

        emb_mode_multiplier = sum([sentence_transformer_config.pooling_mode_cls_token, sentence_transformer_config.pooling_mode_max_tokens,
                                   sentence_transformer_config.pooling_mode_mean_tokens])
        self.classifier = nn.Linear(emb_mode_multiplier * num_vectors_concatenated * self.model_hidden_size, sentence_transformer_config.softmax_num_labels)

        if sentence_transformer_config.triplet_metric == TripletMetric.COSINE:
            self.distance = lambda x, y: 1-F.cosine_similarity(x, y)
        elif sentence_transformer_config.triplet_metric == TripletMetric.EUCLIDEAN:
            self.distance = lambda x, y: F.pairwise_distance(x, y, p=2)
        elif sentence_transformer_config.triplet_metric == TripletMetric.MANHATTAN:
            self.distance = lambda x, y: F.pairwise_distance(x, y, p=1)

    def get_sentence_representation(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Get the embeddings for a batch of inputs

        :param input_ids:
            Tensor of the token embedding ids
        :param token_type_ids:
            Tensor of the segment ids
        :param attention_mask:
            Tensor of the masks
        :return: Tensor of the sentence embeddings for the inputs
        """
        encoded_layers, cls_token = self._get_transformer_output(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        output_vectors = []
        if self.sentence_transformer_config.pooling_mode_cls_token:
            output_vectors.append(cls_token)
        if self.sentence_transformer_config.pooling_mode_max_tokens:
            max_over_time = torch.max(encoded_layers, 1)[0]
            output_vectors.append(max_over_time)
        if self.sentence_transformer_config.pooling_mode_mean_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(encoded_layers.size()).float()
            sum_embeddings = torch.sum(encoded_layers * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            mean = sum_embeddings / sum_mask
            output_vectors.append(mean)

        output_vector = torch.cat(output_vectors, 1)

        return output_vector

    def compute_loss(self, reps: List[Tuple[Tensor]], labels: Tensor = None, alternative_loss: LossFunction = None):
        loss = alternative_loss if alternative_loss is not None else self.sentence_transformer_config.loss_function

        if loss==LossFunction.SOFTMAX:
            # Softmax Classifier
            vectors_concat = []
            rep_a = reps[0]
            rep_b = reps[1]
            if self.sentence_transformer_config.softmax_concatenation_sent_rep:
                vectors_concat.append(rep_a)
                vectors_concat.append(rep_b)

            if self.sentence_transformer_config.softmax_concatenation_sent_difference:
                vectors_concat.append(torch.abs(rep_a - rep_b))

            if self.sentence_transformer_config.softmax_concatenation_sent_multiplication:
                vectors_concat.append(rep_a * rep_b)

            features = torch.cat(vectors_concat, 1)

            output = self.classifier(features)
            loss_fct = nn.CrossEntropyLoss()

            if labels is not None:
                loss = loss_fct(output, labels.view(-1))
                return loss
            else:
                return reps, output
        elif loss==LossFunction.COSINE_SIMILARITY:
            # Cosine Regression
            rep_a = reps[0]
            rep_b = reps[1]
            output = torch.cosine_similarity(rep_a, rep_b)
            loss_fct = nn.MSELoss()

            if labels is not None:
                loss = loss_fct(output, labels.view(-1))
                return loss
            else:
                return reps, output
        elif loss==LossFunction.BATCH_HARD_TRIPLET_LOSS:
            return batch_hard_triplet_loss(labels, reps[0], margin=self.sentence_transformer_config.triplet_margin)
        elif loss==LossFunction.TRIPLET_LOSS:
            rep_anchor, rep_pos, rep_neg = reps[:3]
            distance_pos = self.distance(rep_anchor, rep_pos)
            distance_neg = self.distance(rep_anchor, rep_neg)

            losses = F.relu(distance_pos - distance_neg + self.sentence_transformer_config.triplet_margin)
            return losses.mean()
        elif loss==LossFunction.MULTIPLE_NEGATIVES_RANKING_LOSS:
            reps_a, reps_b = reps[:2]
            return multiple_negatives_ranking_loss(reps_a, reps_b)
