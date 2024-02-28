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
        kl_temperature: float = 1.0,
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
            n_layers_per_step,
            kl_temperature=kl_temperature,
        )
