from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from typing import Any, Iterable, Iterator

import torch
import tqdm
from torch import Tensor, nn
from torch.utils.checkpoint import get_device_states, set_device_states

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer


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
    sentence_features: Iterable[dict[str, Tensor]],
    loss_obj: CachedGISTEmbedLoss,
) -> None:
    """A backward hook to backpropagate the cached gradients mini-batch by mini-batch."""
    assert loss_obj.cache is not None
    assert loss_obj.random_states is not None
    with torch.enable_grad():
        for sentence_feature, grad, random_states in zip(sentence_features, loss_obj.cache, loss_obj.random_states):
            for (reps_mb, _, _), grad_mb in zip(
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


class CachedGISTEmbedLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        guide: SentenceTransformer,
        temperature: float = 0.01,
        mini_batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> None:
        """
        This loss is a combination of :class:`GISTEmbedLoss` and :class:`CachedMultipleNegativesRankingLoss`.
        Typically, :class:`MultipleNegativesRankingLoss` requires a larger batch size for better performance.
        :class:`GISTEmbedLoss` yields stronger training signals than :class:`MultipleNegativesRankingLoss` due to the
        use of a guide model for in-batch negative sample selection. Meanwhile, :class:`CachedMultipleNegativesRankingLoss`
        allows for scaling of the batch size by dividing the computation into two stages of embedding and loss
        calculation, which both can be scaled by mini-batches (https://arxiv.org/pdf/2101.06983.pdf).

        By combining the guided selection from :class:`GISTEmbedLoss` and Gradient Cache from
        :class:`CachedMultipleNegativesRankingLoss`, it is possible to reduce memory usage while maintaining performance
        levels comparable to those of :class:`GISTEmbedLoss`.

        Args:
            model: SentenceTransformer model
            guide: SentenceTransformer model to guide the in-batch negative sample selection.
            temperature: Temperature parameter to scale the cosine similarities.
            mini_batch_size: Mini-batch size for the forward pass, this denotes how much memory is actually used during
                training and evaluation. The larger the mini-batch size, the more memory efficient the training is, but
                the slower the training will be. It's recommended to set it as high as your GPU memory allows. The default
                value is 32.
            show_progress_bar: If True, a progress bar for the mini-batches is shown during training. The default is False.

        References:
            - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://arxiv.org/pdf/1705.00652.pdf
            - Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup: https://arxiv.org/pdf/2101.06983.pdf
            - GISTEmbed: Guided In-sample Selection of Training Negatives for Text Embedding Fine-tuning https://arxiv.org/abs/2402.16829

        Requirements:
            1. (anchor, positive) pairs or (anchor, positive, negative pairs)
            2. Should be used with large batch sizes for superior performance, but has slower training time than :class:`MultipleNegativesRankingLoss`

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (anchor, positive) pairs              | none   |
            +---------------------------------------+--------+
            | (anchor, positive, negative) triplets | none   |
            +---------------------------------------+--------+

        Recommendations:
            - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.training_args.BatchSamplers>`) to
              ensure that no in-batch negatives are duplicates of the anchor or positive samples.

        Relations:
            - Equivalent to :class:`GISTEmbedLoss`, but with caching that allows for much higher batch sizes

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                guide = SentenceTransformer("all-MiniLM-L6-v2")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.CachedGISTEmbedLoss(model, guide, mini_batch_size=64)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.guide = guide
        self.temperature = temperature
        self.similarity_fct = nn.CosineSimilarity(dim=-1)
        if not isinstance(model[0], Transformer) or not isinstance(guide[0], Transformer):
            raise ValueError(
                "Both the training model and the guiding model must be based on the `transformers` architecture."
            )
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mini_batch_size = mini_batch_size
        self.cache: list[list[Tensor]] | None = None
        self.random_states: list[list[RandContext]] | None = None
        self.show_progress_bar = show_progress_bar
        self.must_retokenize = (
            model.tokenizer.vocab != guide.tokenizer.vocab or guide.max_seq_length < model.max_seq_length
        )
        if self.must_retokenize:
            self.tokenizer = model.tokenizer

    def sim_matrix(self, embed1: Tensor, embed2: Tensor) -> Tensor:
        return self.similarity_fct(embed1.unsqueeze(1), embed2.unsqueeze(0))

    def embed_minibatch(
        self,
        sentence_feature: dict[str, Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ) -> tuple[Tensor, RandContext | None]:
        """Do forward pass on a minibatch of the input features and return corresponding embeddings."""
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = {k: v[begin:end] for k, v in sentence_feature.items()}
        with random_state_context:
            with grad_context():
                random_state = RandContext(*sentence_feature_minibatch.values()) if copy_random_state else None
                reps = self.model(sentence_feature_minibatch)["sentence_embedding"]  # (mbsz, hdim)
            with torch.no_grad():
                if self.must_retokenize:
                    decoded = self.tokenizer.batch_decode(
                        sentence_feature_minibatch["input_ids"], skip_special_tokens=True
                    )
                    sentence_feature_minibatch = self.guide.tokenize(decoded)
                    sentence_feature_minibatch = {
                        key: value.to(self.guide.device) for key, value in sentence_feature_minibatch.items()
                    }
                guide_reps = self.guide(sentence_feature_minibatch)["sentence_embedding"]

        return reps, guide_reps, random_state

    def embed_minibatch_iter(
        self,
        sentence_feature: dict[str, Tensor],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
    ) -> Iterator[tuple[Tensor, RandContext | None]]:
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
            reps, guide_reps, random_state = self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=b,
                end=e,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )
            yield reps, guide_reps, random_state  # reps: (mbsz, hdim)

    def calculate_loss_and_cache_gradients(self, reps: list[list[Tensor]], reps_guided: list[list[Tensor]]) -> Tensor:
        """Calculate the cross-entropy loss and cache the gradients wrt. the embeddings."""
        if len(reps) == 2:
            anchor, positive = reps
            anchor_guide, positive_guide = reps_guided
            negative = None
            negative_guide = None
        elif len(reps) == 3:
            anchor, positive, negative = reps
            anchor_guide, positive_guide, negative_guide = reps_guided
        else:
            raise ValueError(f"Expected 2 or 3 embeddings, got {len(reps)}")

        anchor = torch.cat(anchor, dim=0)
        positive = torch.cat(positive, dim=0)
        anchor_guide = torch.cat(anchor_guide, dim=0)
        positive_guide = torch.cat(positive_guide, dim=0)
        # Handle the case where we have a negative sample
        if negative:
            negative = torch.cat(negative, dim=0)
            negative_guide = torch.cat(negative_guide, dim=0)

        labels = torch.arange(anchor.size(0)).long().to(anchor.device)
        batch_size = anchor.shape[0]

        losses: list[torch.Tensor] = []
        for b in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Preparing caches",
            disable=not self.show_progress_bar,
        ):
            e = b + self.mini_batch_size
            # Let's compute the similarity matrices for the combinations of anchor and positive samples.
            guided_ap_sim = self.sim_matrix(anchor_guide[b:e], positive_guide)
            guided_aa_sim = self.sim_matrix(anchor_guide[b:e], anchor_guide)
            guided_pp_sim = self.sim_matrix(positive_guide[b:e], positive_guide)
            # Define the anchor threshold
            guided_sim = guided_ap_sim.diagonal(offset=b).view(-1, 1)

            # Compute similarity scores for current mini-batch.
            # anchor (mbsz,hdim), positive (bsz,hdim)
            ap_sim = self.sim_matrix(anchor[b:e], positive)  # (mbsz,bsz)
            aa_sim = self.sim_matrix(anchor[b:e], anchor)
            pp_sim = self.sim_matrix(positive[b:e], positive)

            # Find which samples cannot be used as negatives because they are
            # more similar to the query than the assigned positive as deemed by the guide model.
            # For these samples, we mask them with -inf to basically ignore their contribution to
            # the loss.
            ap_sim[guided_ap_sim > guided_sim] = -torch.inf
            aa_sim[guided_aa_sim > guided_sim] = -torch.inf
            pp_sim[guided_pp_sim > guided_sim] = -torch.inf

            scores = torch.cat([ap_sim, aa_sim, pp_sim], dim=1)

            # Handle the case where we have a negative sample
            if negative is not None:
                guided_an_sim = self.sim_matrix(anchor_guide[b:e], negative_guide)
                an_sim = self.sim_matrix(anchor[b:e], negative)
                an_sim[guided_an_sim > guided_sim] = -torch.inf
                scores = torch.cat([scores, an_sim], dim=1)
            scores = scores / self.temperature
            loss_mbatch: torch.Tensor = self.cross_entropy_loss(scores, labels[b:e]) * len(scores) / batch_size
            loss_mbatch.backward()
            losses.append(loss_mbatch.detach())

        loss = sum(losses).requires_grad_()

        self.cache = [[r.grad for r in rs] for rs in reps]  # e.g. 3 * bsz/mbsz * (mbsz, hdim)

        return loss

    def calculate_loss(self, reps: list[list[Tensor]], reps_guided: list[list[Tensor]]) -> Tensor:
        """Calculate the cross-entropy loss. No need to cache the gradients."""
        if len(reps) == 2:
            anchor, positive = reps
            anchor_guide, positive_guide = reps_guided
            negative = None
            negative_guide = None
        elif len(reps) == 3:
            anchor, positive, negative = reps
            anchor_guide, positive_guide, negative_guide = reps_guided
        else:
            raise ValueError(f"Expected 2 or 3 embeddings, got {len(reps)}")

        anchor = torch.cat(anchor, dim=0)
        positive = torch.cat(positive, dim=0)
        anchor_guide = torch.cat(anchor_guide, dim=0)
        positive_guide = torch.cat(positive_guide, dim=0)
        # Handle the case where we have a negative sample
        if negative:
            negative = torch.cat(negative, dim=0)
            negative_guide = torch.cat(negative_guide, dim=0)

        labels = torch.arange(anchor.size(0)).long().to(anchor.device)
        batch_size = anchor.shape[0]

        losses: list[torch.Tensor] = []
        for b in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Preparing caches",
            disable=not self.show_progress_bar,
        ):
            e = b + self.mini_batch_size
            # Let's compute the similarity matrices for the combinations of anchor and positive samples.
            guided_ap_sim = self.sim_matrix(anchor_guide[b:e], positive_guide)
            guided_aa_sim = self.sim_matrix(anchor_guide[b:e], anchor_guide)
            guided_pp_sim = self.sim_matrix(positive_guide[b:e], positive_guide)
            # Define the anchor threshold
            guided_sim = guided_ap_sim.diagonal(offset=b).view(-1, 1)

            # Compute similarity scores for current mini-batch.
            # anchor (mbsz,hdim), positive (bsz,hdim)
            ap_sim = self.sim_matrix(anchor[b:e], positive)  # (mbsz,bsz)
            aa_sim = self.sim_matrix(anchor[b:e], anchor)
            pp_sim = self.sim_matrix(positive[b:e], positive)

            # Find which samples cannot be used as negatives because they are
            # more similar to the query than the assigned positive as deemed by the guide model.
            # For these samples, we mask them with -inf to basically ignore their contribution to
            # the loss.
            ap_sim[guided_ap_sim > guided_sim] = -torch.inf
            aa_sim[guided_aa_sim > guided_sim] = -torch.inf
            pp_sim[guided_pp_sim > guided_sim] = -torch.inf

            scores = torch.cat([ap_sim, aa_sim, pp_sim], dim=1)

            # Handle the case where we have a negative sample
            if negative is not None:
                guided_an_sim = self.sim_matrix(anchor_guide[b:e], negative_guide)
                an_sim = self.sim_matrix(anchor[b:e], negative)
                an_sim[guided_an_sim > guided_sim] = -torch.inf
                scores = torch.cat([scores, an_sim], dim=1)
            scores = scores / self.temperature
            loss_mbatch: torch.Tensor = self.cross_entropy_loss(scores, labels[b:e]) * len(scores) / batch_size
            losses.append(loss_mbatch)

        loss = sum(losses)
        return loss

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Step (1): A quick embedding step without gradients/computation graphs to get all the embeddings
        reps = []
        reps_guided = []
        self.random_states = []  # Copy random states to guarantee exact reproduction of the embeddings during the second forward pass, i.e. step (3)
        for sentence_feature in sentence_features:
            reps_mbs = []
            reps_guided_mbs = []
            random_state_mbs = []
            for reps_mb, reps_guided_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=True,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                reps_guided_mbs.append(reps_guided_mb.detach())  # does not requires gradient
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            reps_guided.append(reps_guided_mbs)
            self.random_states.append(random_state_mbs)

        if torch.is_grad_enabled():
            # Step (2): Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings
            loss = self.calculate_loss_and_cache_gradients(reps, reps_guided)

            # Step (3): A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the backward chain
            loss.register_hook(partial(_backward_hook, sentence_features=sentence_features, loss_obj=self))
        else:
            # If grad is not enabled (e.g. in evaluation), then we don't have to worry about the gradients or backward hook
            loss = self.calculate_loss(reps, reps_guided)
        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "guide": self.guide,
            "temperature": self.temperature,
        }
