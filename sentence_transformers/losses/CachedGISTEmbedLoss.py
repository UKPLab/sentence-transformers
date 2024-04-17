from __future__ import annotations
from contextlib import nullcontext
from functools import partial
import torch
from torch import nn, Tensor
from torch.utils.checkpoint import get_device_states, set_device_states
from typing import Iterable, Dict, Iterator, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import tqdm
from sentence_transformers.models import Transformer


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
    loss_obj: CachedGISTEmbedLoss,
):
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
    ):
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

        :param model: SentenceTransformer model
        :param guide: SentenceTransformer model to guide the in-batch negative sample selection.
        :param temperature: Temperature parameter to scale the cosine similarities.

        References:
            - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://arxiv.org/pdf/1705.00652.pdf
            - Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup: https://arxiv.org/pdf/2101.06983.pdf
            - GISTEmbed: Guided In-sample Selection of Training Negatives for Text Embedding Fine-tuning https://arxiv.org/abs/2402.16829

        Requirements:
            1. (anchor, positive) pairs or (anchor, positive, negative pairs)
            2. Should be used with large batch sizes for superior performance, but has slower training time than :class:`MultipleNegativesRankingLoss`

        Relations:
            - Equivalent to :class:`GISTEmbedLoss`, but with caching that allows for much higher batch sizes

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (anchor, positive) pairs              | none   |
            +---------------------------------------+--------+
            | (anchor, positive, negative) triplets | none   |
            +---------------------------------------+--------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, losses, InputExample
                from torch.utils.data import DataLoader

                model = SentenceTransformer('distilbert-base-uncased')
                guide = SentenceTransformer('avsolatorio/GIST-small-Embedding-v0')

                train_examples = [
                    InputExample(texts=['Anchor 1', 'Positive 1']),
                    InputExample(texts=['Anchor 2', 'Positive 2']),
                ]
                train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=1024)  # Here we can try much larger batch sizes!
                train_loss = losses.CachedGISTEmbedLoss(model=model, mini_batch_size=32, guide=guide)
                model.fit(
                    [(train_dataloader, train_loss)],
                    epochs=10,
                )
        """
        super(CachedGISTEmbedLoss, self).__init__()
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
        self.cache: Optional[List[List[Tensor]]] = None
        self.random_states: Optional[List[List[RandContext]]] = None
        self.show_progress_bar = show_progress_bar
        self.must_retokenize = (
            model.tokenizer.vocab != guide.tokenizer.vocab or guide.max_seq_length < model.max_seq_length
        )

    def sim_matrix(self, embed1, embed2):
        return self.similarity_fct(embed1.unsqueeze(1), embed2.unsqueeze(0))

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
            with torch.no_grad():
                if self.must_retokenize:
                    decoded = self.model.tokenizer.batch_decode(
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
            reps, guide_reps, random_state = self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=b,
                end=e,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )
            yield reps, guide_reps, random_state  # reps: (mbsz, hdim)

    def calculate_loss_and_cache_gradients(self, reps: List[List[Tensor]], reps_guided: List[List[Tensor]]) -> Tensor:
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
            raise ValueError("Expected 2 or 3 embeddings, got {}".format(len(reps)))

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

        losses: List[torch.Tensor] = []
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

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor) -> Tensor:
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

        # Step (2): Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings
        loss = self.calculate_loss_and_cache_gradients(reps, reps_guided)
        # Step (3): A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the backward chain
        loss.register_hook(partial(_backward_hook, sentence_features=sentence_features, loss_obj=self))
        return loss

    def get_config_dict(self):
        return {
            "guide": self.guide,
            "temperature": self.temperature,
        }
