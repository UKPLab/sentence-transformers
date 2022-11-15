from __future__ import annotations
from contextlib import nullcontext
from functools import partial
import torch
from torch import nn, Tensor
from typing import Iterable, Dict, Iterator, List, Optional
from sentence_transformers import SentenceTransformer
from sentence_transformers import util


def _hook(
    grad_output: Tensor,
    sentence_features: Iterable[Dict[str, Tensor]],
    loss: CachedMultipleNegativesRankingLoss,
):
    """Do the actual backward pass minibatch by minibatch with the cached gradients."""
    assert loss.cache is not None
    with torch.enable_grad():
        for sentence_feature, grad in zip(sentence_features, loss.cache):
            for reps_mb, grad_mb in zip(
                loss.forward_minibatch_iter(sentence_feature, True), grad
            ):
                surrogate = (
                    torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
                )
                surrogate.backward()


class CachedMultipleNegativesRankingLoss(nn.Module):
    """
    Boosted version of MultipleNegativesRankingLoss (https://arxiv.org/pdf/1705.00652.pdf) by GradCache (https://arxiv.org/pdf/2101.06983.pdf).

    Constrastive learning (here our MNRL loss) with in-batch negatives is usually hard to work with large batch sizes due to (GPU) memory limitation.
    Even with batch-scaling methods like gradient-scaling, it cannot work either. This is because the in-batch negatives make the data points within
    the same batch non-independent and thus the batch cannot be broke down into mini-batches. GradCache is a smart way to solve this problem.
    It achieves the goal by dividing the computation into two stages of embedding and loss calculation, which both can be scaled by mini-batches.
    As a result, memory of constant size (e.g. that works with batch size = 32) can now process much larger batches (e.g. 65536).

    In detail:
        (1) It first does a quick embedding step without gradients/computation graphs to get all the embeddings;
        (2) Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings;
        (3) A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the backward chain.

    Notes: All steps are done with mini-batches. In the original implementation of GradCache, (2) is not done in mini-batches and
    requires a lot memory when batch size large. One drawback is about the speed. GradCache will sacrifice around 20% computation time according to the paper.

    Example:

        from sentence_transformers import SentenceTransformer
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=1024)  # Here we can try much larger batch sizes!
        train_loss = losses.CachedMultipleNegativesRankingLoss(model=model, mini_batch_size: int = 32)
    """

    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        similarity_fct: callable[[Tensor, Tensor], Tensor] = util.cos_sim,
        mini_batch_size: int = 32,
    ):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(CachedMultipleNegativesRankingLoss, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.mini_batch_size = mini_batch_size
        self.cache: Optional[List[List[Tensor]]] = None

    def forward_minibatch(
        self, sentence_feature: Dict[str, Tensor], begin: int, end: int, with_grad: bool
    ) -> Tensor:
        """Do forward pass on a minibatch of the input features and return corresponding embeddings."""
        grad_context = nullcontext if with_grad else torch.no_grad
        sentence_feature_minibatch = {
            k: v[begin:end] for k, v in sentence_feature.items()
        }
        with grad_context():
            reps = self.model(sentence_feature_minibatch)[
                "sentence_embedding"
            ]  # (mbsz, hdim)
        return reps

    def forward_minibatch_iter(
        self, sentence_feature: Dict[str, Tensor], with_grad: bool
    ) -> Iterator[Tensor]:
        """Do forward pass on all the minibatches of the input features and yield corresponding embeddings."""
        input_ids: Tensor = sentence_feature["input_ids"]
        bsz, _ = input_ids.shape
        for b in range(0, bsz, self.mini_batch_size):
            e = b + self.mini_batch_size
            reps = self.forward_minibatch(sentence_feature, b, e, with_grad)
            yield reps  # (mbsz, hdim)

    def forward(
        self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor
    ) -> Tensor:
        reps = [
            [
                reps_mb.detach().requires_grad_()
                for reps_mb in self.forward_minibatch_iter(sentence_feature, False)
            ]
            for sentence_feature in sentence_features
        ]
        embeddings_a = torch.cat(reps[0])  # (bsz, hdim)
        embeddings_b = torch.cat(
            [torch.cat(r) for r in reps[1:]]
        )  # ((1 + nneg) * bsz, hdim)

        batch_size = len(embeddings_a)
        labels = torch.tensor(
            range(batch_size), dtype=torch.long, device=embeddings_a.device
        )  # (bsz, (1 + nneg) * bsz)  Example a[i] should match with b[i]
        losses: List[torch.Tensor] = []
        for b in range(0, batch_size, self.mini_batch_size):
            e = b + self.mini_batch_size
            scores: Tensor = (
                self.similarity_fct(embeddings_a[b:e], embeddings_b) * self.scale
            )
            loss_mbatch = (
                nn.functional.cross_entropy(scores, labels[b:e])
                * len(scores)
                / batch_size
            )
            losses.append(loss_mbatch)

        loss = sum(losses)
        loss.backward()
        loss = loss.detach().requires_grad_()

        self.cache = [
            [r.grad for r in rs] for rs in reps
        ]  # e.g. 3 * bsz/mbsz * (mbsz, hdim)

        loss.register_hook(
            partial(_hook, sentence_features=sentence_features, loss=self)
        )
        return loss

    def get_config_dict(self):
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}
