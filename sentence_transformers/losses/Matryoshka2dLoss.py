from typing import Any, Dict, List, Optional, Union
from torch.nn import Module
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.losses import AdaptiveLayerLoss, MatryoshkaLoss


class Matryoshka2dLoss(AdaptiveLayerLoss):
    def __init__(
        self,
        model: SentenceTransformer,
        loss: Module,
        matryoshka_dims: List[int],
        matryoshka_weights: Optional[List[Union[float, int]]] = None,
        n_layers_per_step: int = 1,
        n_dims_per_step: int = 1,
        last_layer_weight: float = 1.0,
        prior_layers_weight: float = 1.0,
        kl_div_weight: float = 1.0,
        kl_temperature: float = 0.3,
    ) -> None:
        """
        The Matryoshka2dLoss can be seen as a loss *modifier* that combines the :class:`AdaptiveLayerLoss` and the
        :class:`MatryoshkaLoss`. This allows you to train an embedding model that 1) allows users to specify the number
        of model layers to use, and 2) allows users to specify the output dimensions to use.

        The former is useful for when you want users to have the option to lower the number of layers used to improve
        their inference speed and memory usage, and the latter is useful for when you want users to have the option to
        lower the output dimensions to improve the efficiency of their downstream tasks (e.g. retrieval) or to lower
        their storage costs.

        Note, this uses `n_layers_per_step=1` and `n_dims_per_step=1` as default, following the original 2DMSE
        implementation.

        :param model: SentenceTransformer model
        :param loss: The loss function to be used, e.g. :class:`MultipleNegativesRankingLoss`, :class:`CoSENTLoss`, etc.
        :param matryoshka_dims: A list of embedding dimensions to be used for the loss function, e.g. [768, 512, 256, 128, 64].
        :param matryoshka_weights: A list of weights to be used for the loss function, e.g. [1, 1, 1, 1, 1]. If None, then the
            weights will be set to 1 for all dimensions.
        :param n_layers_per_step: The number of layers to use per step. If -1, then all layers are used. If > 0, then
            a random sample of n_layers_per_step layers are used per step. The 2DMSE paper uses `n_layers_per_step=1`.
            The default value is -1.
        :param n_dims_per_step: The number of dimensions to use per step. If -1, then all dimensions are used. If > 0, then
            a random sample of n_dims_per_step dimensions are used per step. The default value is -1.
        :param last_layer_weight: The weight to use for the loss of the final layer. Increase this to focus more on the
            performance when using all layers. The default value is 1.0.
        :param prior_layers_weight: The weight to use for the loss of the prior layers. Increase this to focus more on
            the performance when using fewer layers. The default value is 1.0.
        :param kl_div_weight: The weight to use for the KL-divergence loss that is used to make the prior layers match
            that of the last layer. Increase this to focus more on the performance when using fewer layers. The default
            value is 1.0.
        :param kl_temperature: The temperature to use for the KL-divergence loss. If 0, then the KL-divergence loss is
            not used. The default value is 1.0.

        References:
            - See the 2D Matryoshka Sentence Embeddings (2DMSE) paper: https://arxiv.org/abs/2402.14776
            - `Matryoshka Embeddings <../../examples/training/matryoshka/README.html>`_
            - `Adaptive Layers <../../examples/training/adaptive_layer/README.html>`_

        Requirements:
            1. The base loss cannot be :class:`CachedMultipleNegativesRankingLoss`.

        Relations:
            - :class:`MatryoshkaLoss` is used in this loss, and it is responsible for the dimensionality reduction.
            - :class:`AdaptiveLayerLoss` is used in this loss, and it is responsible for the layer reduction.

        Input:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | any                                   | any    |
            +---------------------------------------+--------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, losses, InputExample
                from torch.utils.data import DataLoader

                model = SentenceTransformer('microsoft/mpnet-base')
                train_examples = [
                    InputExample(texts=['Anchor 1', 'Positive 1']),
                    InputExample(texts=['Anchor 2', 'Positive 2']),
                ]
                train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
                train_loss = losses.MultipleNegativesRankingLoss(model=model)
                train_loss = losses.Matryoshka2dLoss(model, train_loss, [768, 512, 256, 128, 64])
                model.fit(
                    [(train_dataloader, train_loss)],
                    epochs=10,
                )
        """
        matryoshka_loss = MatryoshkaLoss(
            model,
            loss,
            matryoshka_dims,
            matryoshka_weights=matryoshka_weights,
            n_dims_per_step=n_dims_per_step,
        )
        super().__init__(
            model,
            matryoshka_loss,
            n_layers_per_step=n_layers_per_step,
            last_layer_weight=last_layer_weight,
            prior_layers_weight=prior_layers_weight,
            kl_div_weight=kl_div_weight,
            kl_temperature=kl_temperature,
        )

    def get_config_dict(self) -> Dict[str, Any]:
        return {
            **super().get_config_dict(),
            **self.loss.get_config_dict(),
        }
