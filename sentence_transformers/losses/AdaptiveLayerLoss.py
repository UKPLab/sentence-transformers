import random
from typing import Dict, Iterable, List, Tuple
import warnings
from torch import Tensor, nn
from torch.nn import functional as F
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses.CachedMultipleNegativesRankingLoss import CachedMultipleNegativesRankingLoss
from sentence_transformers.models import Transformer


class TransformerDecorator:
    def __init__(self, transformer: Transformer, original_forward):
        self.transformer = transformer
        self.original_forward = original_forward
        self.embeddings: List[Tuple[Tensor]] = []
        self.last_embeddings: List[Tensor] = []
        self.features: List[Dict[str, Tensor]] = []
        self.layer_idx = None
        self.call_idx = 0

    def set_layer_idx(self, layer_idx):
        self.layer_idx = layer_idx
        self.call_idx = 0

    def get_layer_embeddings(self):
        return torch.concat([embedding[self.layer_idx] for embedding in self.embeddings], dim=1)

    def __call__(self, features):
        if self.layer_idx is None:
            output = self.call_grow_cache(features)
        else:
            output = self.call_use_cache(features)
            self.call_idx += 1
        return output

    def call_grow_cache(self, features):
        """
        Temporarily sets the output_hidden_states to True, runs the model, and then restores the original setting.
        Use the all_layer_embeddings to get the embeddings of all layers.
        """
        original_output_hidden_states = self.transformer.auto_model.config.output_hidden_states
        self.transformer.auto_model.config.output_hidden_states = True

        output = self.original_forward(features)
        # We ignore the first layer, as it is the input embeddings
        # and the last layer, as we already computed the loss over it
        self.num_layers = len(output["all_layer_embeddings"]) - 1
        self.embeddings.append(output["all_layer_embeddings"][1:-1])
        self.last_embeddings.append(output["token_embeddings"])
        self.features.append(
            {key: value for key, value in output.items() if key not in ["all_layer_embeddings", "token_embeddings"]}
        )

        # Restore original setting
        self.transformer.auto_model.config.output_hidden_states = original_output_hidden_states

        if original_output_hidden_states:
            del output["all_layer_embeddings"]

        return output

    def call_use_cache(self, features):
        return {**self.features[self.call_idx], "token_embeddings": self.embeddings[self.call_idx][self.layer_idx]}


class ForwardDecorator:
    def __init__(self, fn):
        self.fn = fn
        self.embeddings = []

    def __call__(self, features):
        output = self.fn(features)
        self.embeddings.append(output["sentence_embedding"])
        return output

    def get_embeddings(self):
        embeddings = torch.concat(self.embeddings, dim=0)
        self.embeddings = []
        return embeddings


class AdaptiveLayerLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        loss: nn.Module,
        n_layers_per_step: int = -1,
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.n_layers_per_step = n_layers_per_step
        assert isinstance(self.model[0], Transformer)
        if isinstance(loss, CachedMultipleNegativesRankingLoss):
            warnings.warn("MatryoshkaLoss is not compatible with CachedMultipleNegativesRankingLoss.", stacklevel=2)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Decorate the forward function of the transformer to cache the embeddings of all layers
        original_transformer_forward = self.model[0].forward
        transformer_decorator = TransformerDecorator(self.model[0], original_transformer_forward)
        self.model[0].forward = transformer_decorator

        # Decorate the forward function of the model to get the embeddings after all modules (e.g. pooling)
        original_forward = self.model.forward
        forward_decorator = ForwardDecorator(original_forward)
        self.model.forward = forward_decorator

        # Run the loss normally: i.e. the final layer, but 1) use the transformers decorator to cache
        # the embeddings of all layers and 2) use the forward decorator to get the embeddings after all modules
        # for the KL-divergence loss
        loss = self.loss(sentence_features, labels)
        final_embeddings = forward_decorator.get_embeddings()
        final_embeddings = F.softmax(final_embeddings, dim=-1)

        num_layers = transformer_decorator.num_layers
        layer_indices = range(num_layers - 1)
        if self.n_layers_per_step > 0 and self.n_layers_per_step < num_layers - 1:
            layer_indices = random.sample(layer_indices, self.n_layers_per_step)

        # This loop is over `num_layer - 1` layers because we already computed the loss over the final layer
        for layer_idx in layer_indices:
            # Add regular loss for each layer by using the cached embeddings of that layer
            transformer_decorator.set_layer_idx(layer_idx)
            layer_loss = self.loss(sentence_features, labels)
            loss = loss + layer_loss / len(layer_indices)

            # and KL-divergence loss between the current layer and the final layer
            # Note: we use "batchmean" reduction as that aligns with the mathematical definition
            embeddings = forward_decorator.get_embeddings()
            kl_div_loss = F.kl_div(F.log_softmax(embeddings, dim=-1), final_embeddings, reduction="batchmean")
            loss = loss + kl_div_loss / len(layer_indices)

        self.model[0].forward = original_transformer_forward
        self.model.forward = original_forward

        return loss
