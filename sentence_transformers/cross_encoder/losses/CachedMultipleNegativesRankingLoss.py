from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from functools import partial

import torch
import tqdm
from torch import Tensor, nn
from torch.utils.checkpoint import get_device_states, set_device_states

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.losses import MultipleNegativesRankingLoss


class RandContext:
    """
    Random-state context manager class. Reference: https://github.com/luyug/GradCache.

    This class will back up the pytorch's random state during initialization. Then when the context is activated,
    the class will set up the random state with the backed-up one.
    """

    def __init__(self, *tensors) -> None:
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self) -> None:
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


def _backward_hook(
    grad_output: Tensor,
    pairs: list[list[str]],
    loss_obj: CachedMultipleNegativesRankingLoss,
) -> None:
    """A backward hook to backpropagate the cached gradients mini-batch by mini-batch."""
    assert loss_obj.cache is not None
    assert loss_obj.random_states is not None
    with torch.enable_grad():
        # for sentence_feature, grad, random_states in zip(pairs, loss_obj.cache, loss_obj.random_states):
        for (minibatch_logits, _), minibatch_grad in zip(
            loss_obj.predict_minibatch_iter(
                pairs=pairs,
                with_grad=True,
                copy_random_state=False,
                random_states=loss_obj.random_states,
            ),
            loss_obj.cache,
        ):
            surrogate = torch.dot(minibatch_logits.flatten(), minibatch_grad.flatten()) * grad_output
            surrogate.backward()


class CachedMultipleNegativesRankingLoss(MultipleNegativesRankingLoss):
    def __init__(
        self,
        model: CrossEncoder,
        num_negatives: int | None = 4,
        scale: float = 20.0,
        activation_fct: nn.Module | None = nn.Tanh(),
        mini_batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> None:
        super().__init__(model, num_negatives, scale, activation_fct)
        self.mini_batch_size = mini_batch_size
        self.show_progress_bar = show_progress_bar

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.cache: list[list[Tensor]] | None = None
        self.random_states: list[list[RandContext]] | None = None

        if self.model.num_labels != 1:
            raise ValueError(
                f"{self.__class__.__name__} expects a model with 1 output label, "
                f"but got a model with {self.model.num_labels} output labels."
            )

    def predict_minibatch(
        self,
        pairs: list[list[str]],
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ) -> tuple[Tensor, RandContext | None]:
        """Do forward pass on a minibatch of the input features and return corresponding embeddings."""
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        with random_state_context:
            with grad_context():
                random_state = RandContext(pairs) if copy_random_state else None
                logits = self.call_model_with_pairs(pairs)
        return logits, random_state

    def predict_minibatch_iter(
        self,
        pairs: list[list[str]],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
    ) -> Iterator[tuple[Tensor, RandContext | None]]:
        """Do forward pass on all the minibatches of the input features and yield corresponding embeddings."""
        for i, b in enumerate(
            tqdm.trange(
                0,
                len(pairs),
                self.mini_batch_size,
                desc="Predict mini-batches",
                disable=not self.show_progress_bar,
            )
        ):
            e = b + self.mini_batch_size
            mini_batch_pairs = pairs[b:e]

            logits, random_state = self.predict_minibatch(
                pairs=mini_batch_pairs,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )
            yield logits, random_state  # reps: (mbsz, hdim)

    def calculate_loss_and_cache_gradients(self, logits: list[Tensor], batch_size: int) -> Tensor:
        """Calculate the cross-entropy loss and cache the gradients wrt. the embeddings."""
        loss = self.calculate_loss(logits, batch_size)
        loss.backward()
        loss = loss.detach().requires_grad_()

        self.cache = [logit.grad for logit in logits]

        return loss

    def forward(self, inputs: list[list[str]], labels: Tensor) -> Tensor:
        # Step (1): A quick embedding step without gradients/computation graphs to get all the embeddings
        anchors = inputs[0][::]
        candidates = inputs[1][::]
        batch_size = len(anchors)

        # In-batch negatives:
        for negatives in self.get_in_batch_negatives(inputs[0], inputs[1:]):
            anchors.extend(inputs[0])
            candidates.extend(negatives)

        # Hard negatives:
        for negatives in inputs[2:]:
            anchors.extend(inputs[0])
            candidates.extend(negatives)

        pairs = list(zip(anchors, candidates))

        logits = []
        self.random_states = []
        for minibatch_logits, random_state in self.predict_minibatch_iter(
            pairs=pairs,
            with_grad=False,
            copy_random_state=True,
        ):
            logits.append(minibatch_logits.detach().requires_grad_())
            self.random_states.append(random_state)

        if torch.is_grad_enabled():
            # Step (2): Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings
            loss = self.calculate_loss_and_cache_gradients(logits, batch_size)

            # Step (3): A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the backward chain
            loss.register_hook(partial(_backward_hook, pairs=pairs, loss_obj=self))
        else:
            # If grad is not enabled (e.g. in evaluation), then we don't have to worry about the gradients or backward hook
            loss = self.calculate_loss(logits, batch_size)

        return loss
