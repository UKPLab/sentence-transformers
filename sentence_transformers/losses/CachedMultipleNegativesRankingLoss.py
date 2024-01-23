from __future__ import annotations
from contextlib import nullcontext
from functools import partial
import torch
from torch import nn, Tensor
from torch.utils.checkpoint import get_device_states, set_device_states
from typing import Iterable, Dict, Iterator, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import tqdm


class RandContext:
    """
    Random-state context manager class. Reference: https://github.com/luyug/GradCache.

    This class will back up the pytorch's random state during initialization. Then when the context is activated,
    the class will set up the random state with the backed-up one.
    """

    def __init__(self, *tensors):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


def _backward_hook(
    grad_output: Tensor,
    sentence_features: Iterable[Dict[str, Tensor]],
    loss_obj: CachedMultipleNegativesRankingLoss,
):
    """A backward hook to backpropagate the cached gradients mini-batch by mini-batch."""
    assert loss_obj.cache is not None
    assert loss_obj.random_states is not None
    with torch.enable_grad():
        for sentence_feature, grad, random_states in zip(sentence_features, loss_obj.cache, loss_obj.random_states):
            for (reps_mb, _), grad_mb in zip(
                loss_obj.embed_minibatch_iter(
                    sentence_feature=sentence_feature,
                    with_grad=True,
                    copy_random_state=False,
                    random_states=random_states,
                ),
                grad,
            ):
                surrogate = torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
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

    Example::

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
        show_progress_bar: bool = False,
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
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mini_batch_size = mini_batch_size
        self.cache: Optional[List[List[Tensor]]] = None
        self.random_states: Optional[List[List[RandContext]]] = None
        self.show_progress_bar = show_progress_bar

    def embed_minibatch(
        self,
        sentence_feature: Dict[str, Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: Optional[RandContext] = None,
    ) -> Tuple[Tensor, Optional[RandContext]]:
        """Do forward pass on a minibatch of the input features and return corresponding embeddings."""
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = {k: v[begin:end] for k, v in sentence_feature.items()}
        with random_state_context:
            with grad_context():
                random_state = RandContext(*sentence_feature_minibatch.values()) if copy_random_state else None
                reps = self.model(sentence_feature_minibatch)["sentence_embedding"]  # (mbsz, hdim)
        return reps, random_state

    def embed_minibatch_iter(
        self,
        sentence_feature: Dict[str, Tensor],
        with_grad: bool,
        copy_random_state: bool,
        random_states: Optional[List[RandContext]] = None,
    ) -> Iterator[Tuple[Tensor, Optional[RandContext]]]:
        """Do forward pass on all the minibatches of the input features and yield corresponding embeddings."""
        input_ids: Tensor = sentence_feature["input_ids"]
        bsz, _ = input_ids.shape
        for i, b in enumerate(
            tqdm.trange(
                0,
                bsz,
                self.mini_batch_size,
                desc="Embed mini-batches",
                disable=not self.show_progress_bar,
            )
        ):
            e = b + self.mini_batch_size
            reps, random_state = self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=b,
                end=e,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )
            yield reps, random_state  # reps: (mbsz, hdim)

    def calculate_loss_and_cache_gradients(self, reps: List[List[Tensor]]) -> Tensor:
        """Calculate the cross-entropy loss and cache the gradients wrt. the embeddings."""
        embeddings_a = torch.cat(reps[0])  # (bsz, hdim)
        embeddings_b = torch.cat([torch.cat(r) for r in reps[1:]])  # ((1 + nneg) * bsz, hdim)

        batch_size = len(embeddings_a)
        labels = torch.tensor(
            range(batch_size), dtype=torch.long, device=embeddings_a.device
        )  # (bsz, (1 + nneg) * bsz)  Example a[i] should match with b[i]
        losses: List[torch.Tensor] = []
        for b in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Preparing caches",
            disable=not self.show_progress_bar,
        ):
            e = b + self.mini_batch_size
            scores: Tensor = self.similarity_fct(embeddings_a[b:e], embeddings_b) * self.scale
            loss_mbatch: torch.Tensor = self.cross_entropy_loss(scores, labels[b:e]) * len(scores) / batch_size
            loss_mbatch.backward()
            losses.append(loss_mbatch.detach())

        loss = sum(losses).requires_grad_()

        self.cache = [[r.grad for r in rs] for rs in reps]  # e.g. 3 * bsz/mbsz * (mbsz, hdim)

        return loss

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Step (1): A quick embedding step without gradients/computation graphs to get all the embeddings
        reps = []
        self.random_states = []  # Copy random states to guarantee exact reproduction of the embeddings during the second forward pass, i.e. step (3)
        for sentence_feature in sentence_features:
            reps_mbs = []
            random_state_mbs = []
            for reps_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=True,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            self.random_states.append(random_state_mbs)

        # Step (2): Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings
        loss = self.calculate_loss_and_cache_gradients(reps)

        # Step (3): A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the backward chain
        loss.register_hook(partial(_backward_hook, sentence_features=sentence_features, loss_obj=self))
        return loss

    def get_config_dict(self):
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}
