from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from functools import partial
from typing import Any, Literal

import torch
import tqdm
from torch import Tensor, nn
from torch.utils.checkpoint import get_device_states, set_device_states

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding, Transformer


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
        margin_strategy: Literal["absolute", "relative"] = "absolute",
        margin: float = 0.0,
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

        You can apply different false-negative filtering strategies to discard hard negatives that are too similar to
        the positive. Two strategies are supported:

            - "absolute": Discards negatives whose similarity score is greater than or equal to ``positive_score - margin``.
            - "relative": Discards negatives whose similarity score is greater than or equal to ``positive_score * (1 - margin)``.

        Args:
            model: SentenceTransformer model
            guide: SentenceTransformer model to guide the in-batch negative sample selection.
            temperature: Temperature parameter to scale the cosine similarities.
            mini_batch_size: Mini-batch size for the forward pass, this denotes how much memory is actually used during
                training and evaluation. The larger the mini-batch size, the more memory efficient the training is, but
                the slower the training will be. It's recommended to set it as high as your GPU memory allows. The default
                value is 32.
            show_progress_bar: If True, a progress bar for the mini-batches is shown during training. The default is False.
            margin_strategy: Strategy used for false negative filtering. One of {"absolute", "relative"}.
            margin: The margin value for filtering negatives. Defaults to 0.0, together with the "absolute" strategy,
                this only removes negatives that are more similar to the query than the positive is to the query.

        References:
            - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://arxiv.org/pdf/1705.00652.pdf
            - Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup: https://arxiv.org/pdf/2101.06983.pdf
            - GISTEmbed: Guided In-sample Selection of Training Negatives for Text Embedding Fine-tuning https://arxiv.org/abs/2402.16829

        Requirements:
            1. (anchor, positive) pairs or (anchor, positive, negative pairs)
            2. Should be used with large batch sizes for superior performance, but has slower training time than :class:`MultipleNegativesRankingLoss`

        Inputs:
            +-------------------------------------------------+--------+
            | Texts                                           | Labels |
            +=================================================+========+
            | (anchor, positive) pairs                        | none   |
            +-------------------------------------------------+--------+
            | (anchor, positive, negative) triplets           | none   |
            +-------------------------------------------------+--------+
            | (anchor, positive, negative_1, ..., negative_n) | none   |
            +-------------------------------------------------+--------+

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
                loss = losses.CachedGISTEmbedLoss(
                    model,
                    guide,
                    mini_batch_size=64,
                    margin_strategy="absolute",   # or "relative" (e.g., margin=0.05 for max. 95% of positive similarity)
                    margin=0.1
                )

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        if isinstance(model[0], StaticEmbedding):
            raise ValueError(
                "CachedGISTEmbedLoss is not compatible with a SentenceTransformer model based on a StaticEmbedding. "
                "Consider using GISTEmbedLoss instead."
            )
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
        if margin_strategy not in ("absolute", "relative"):
            raise ValueError("margin_strategy must be 'absolute' or 'relative'.")
        self.margin_strategy = margin_strategy
        self.margin = margin

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
    ) -> tuple[Tensor, Tensor, RandContext | None]:
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
    ) -> Iterator[tuple[Tensor, Tensor, RandContext | None]]:
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
        """Generalized function to calculate the cross-entropy loss and cache the gradients wrt. the embeddings."""
        loss = self.calculate_loss(reps, reps_guided, with_backward=True)
        loss = loss.detach().requires_grad_()

        self.cache = [[r.grad for r in rs] for rs in reps]

        return loss

    def calculate_loss(
        self, reps: list[list[Tensor]], reps_guided: list[list[Tensor]], with_backward: bool = False
    ) -> Tensor:
        """Generalized function to calculate the cross-entropy loss without caching gradients."""
        if len(reps) != len(reps_guided):
            raise ValueError("reps and reps_guided must have the same length")

        # Concatenate embeddings along the batch dimension
        concatenated_reps = [torch.cat(rep, dim=0) for rep in reps]
        concatenated_guided_reps = [torch.cat(rep_guide, dim=0) for rep_guide in reps_guided]

        labels = torch.arange(concatenated_reps[0].size(0)).long().to(concatenated_reps[0].device)
        batch_size = concatenated_reps[0].shape[0]

        losses: list[torch.Tensor] = []
        for b in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Calculating loss",
            disable=not self.show_progress_bar,
        ):
            e = b + self.mini_batch_size

            # Compute guided similarity matrices for anchor-positive, anchor-anchor, and positive-positive samples
            guided_ap_sim = self.sim_matrix(concatenated_guided_reps[0][b:e], concatenated_guided_reps[1])
            guided_aa_sim = self.sim_matrix(concatenated_guided_reps[0][b:e], concatenated_guided_reps[0])
            guided_pp_sim = self.sim_matrix(concatenated_guided_reps[1][b:e], concatenated_guided_reps[1])

            # Define the anchor threshold for each similarity matrix
            guided_sim = guided_ap_sim.diagonal(offset=b).view(-1, 1)

            # Compute similarity scores for the current mini-batch
            ap_sim = self.sim_matrix(concatenated_reps[0][b:e], concatenated_reps[1])  # anchor-positive similarity
            aa_sim = self.sim_matrix(concatenated_reps[0][b:e], concatenated_reps[0])  # anchor-anchor similarity
            pp_sim = self.sim_matrix(concatenated_reps[1][b:e], concatenated_reps[1])  # positive-positive similarity

            # This uses guided (teacher) similarity as a dynamic threshold to identify and suppress false negatives
            def mask_false_negatives(guided_sim_mat, sim_mat, positive_mask: Tensor | None = None):
                if self.margin_strategy == "absolute":
                    # Remove samples whose guided similarity is higher than (positive_sim - margin)
                    mask = guided_sim_mat > (guided_sim - self.margin)
                elif self.margin_strategy == "relative":
                    # Remove samples whose guided similarity is higher than (positive_sim * margin)
                    mask = guided_sim_mat > (guided_sim * (1 - self.margin))

                if positive_mask is not None:
                    # Ensure true positive pairs are not masked out
                    mask = mask & ~positive_mask
                sim_mat[mask] = -torch.inf
                return sim_mat

            # Create a mask to protect true positive pairs in the anchor-positive matrix (i.e., diagonal elements)
            positive_mask = torch.eye(*guided_ap_sim.shape, dtype=torch.bool, device=guided_ap_sim.device)
            positive_mask = positive_mask.roll(b)

            # Apply false negative suppression to each similarity matrix using guided similarity as anchor
            ap_sim = mask_false_negatives(guided_ap_sim, ap_sim, positive_mask=positive_mask)  # anchor-positive
            aa_sim = mask_false_negatives(guided_aa_sim, aa_sim)  # anchor-anchor
            pp_sim = mask_false_negatives(guided_pp_sim, pp_sim)  # positive-positive

            # Concatenate the similarity matrices for anchor-positive, anchor-anchor, and positive-positive
            scores = torch.cat([ap_sim, aa_sim, pp_sim], dim=1)

            # If there are negatives (len(reps) > 2), process them
            if len(concatenated_reps) > 2:
                for i in range(2, len(concatenated_reps)):  # Start from 2 since first 2 are anchor-positive
                    guided_neg_sim = self.sim_matrix(concatenated_guided_reps[0][b:e], concatenated_guided_reps[i])
                    neg_sim = self.sim_matrix(concatenated_reps[0][b:e], concatenated_reps[i])
                    neg_sim = mask_false_negatives(guided_neg_sim, neg_sim)
                    scores = torch.cat([scores, neg_sim], dim=1)

            # Normalize the scores and calculate the cross-entropy loss
            scores = scores / self.temperature
            loss_mbatch: torch.Tensor = self.cross_entropy_loss(scores, labels[b:e]) * len(scores) / batch_size
            if with_backward:
                loss_mbatch.backward()
                loss_mbatch = loss_mbatch.detach()
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
            "mini_batch_size": self.mini_batch_size,
            "margin_strategy": self.margin_strategy,
            "margin": self.margin,
        }
