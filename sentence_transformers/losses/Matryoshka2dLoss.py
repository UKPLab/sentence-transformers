from typing import List, Optional, Union
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
        # Note, this uses n_layers_per_step=1 & n_dims_per_step=1 as default, following the 2DMSE implementation
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
