from __future__ import annotations

import functools
import heapq
import importlib
import logging
import os
import queue
import random
import sys
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, metadata
from typing import TYPE_CHECKING, Any, Callable, Literal, overload

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from torch import Tensor, device
from tqdm import trange
from tqdm.autonotebook import tqdm
from transformers import is_torch_npu_available

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from datasets import Dataset

    from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder
    from sentence_transformers.SentenceTransformer import SentenceTransformer


def _convert_to_tensor(a: list | np.ndarray | Tensor) -> Tensor:
    """
    Converts the input `a` to a PyTorch tensor if it is not already a tensor.

    Args:
        a (Union[list, np.ndarray, Tensor]): The input array or tensor.

    Returns:
        Tensor: The converted tensor.
    """
    if not isinstance(a, Tensor):
        a = torch.tensor(a)
    return a


def _convert_to_batch(a: Tensor) -> Tensor:
    """
    If the tensor `a` is 1-dimensional, it is unsqueezed to add a batch dimension.

    Args:
        a (Tensor): The input tensor.

    Returns:
        Tensor: The tensor with a batch dimension.
    """
    if a.dim() == 1:
        a = a.unsqueeze(0)
    return a


def _convert_to_batch_tensor(a: list | np.ndarray | Tensor) -> Tensor:
    """
    Converts the input data to a tensor with a batch dimension.

    Args:
        a (Union[list, np.ndarray, Tensor]): The input data to be converted.

    Returns:
        Tensor: The converted tensor with a batch dimension.
    """
    a = _convert_to_tensor(a)
    a = _convert_to_batch(a)
    return a


def pytorch_cos_sim(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the cosine similarity between two tensors.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Matrix with res[i][j] = cos_sim(a[i], b[j])
    """
    return cos_sim(a, b)


def cos_sim(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor) -> Tensor:
    """
    Computes the cosine similarity between two tensors.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Matrix with res[i][j] = cos_sim(a[i], b[j])
    """
    a = _convert_to_batch_tensor(a)
    b = _convert_to_batch_tensor(b)

    a_norm = normalize_embeddings(a)
    b_norm = normalize_embeddings(b)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def pairwise_cos_sim(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the pairwise cosine similarity cos_sim(a[i], b[i]).

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Vector with res[i] = cos_sim(a[i], b[i])
    """
    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)

    return pairwise_dot_score(normalize_embeddings(a), normalize_embeddings(b))


def dot_score(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor) -> Tensor:
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Matrix with res[i][j] = dot_prod(a[i], b[j])
    """
    a = _convert_to_batch_tensor(a)
    b = _convert_to_batch_tensor(b)

    return torch.mm(a, b.transpose(0, 1))


def pairwise_dot_score(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the pairwise dot-product dot_prod(a[i], b[i]).

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Vector with res[i] = dot_prod(a[i], b[i])
    """
    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)

    return (a * b).sum(dim=-1)


def manhattan_sim(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor) -> Tensor:
    """
    Computes the manhattan similarity (i.e., negative distance) between two tensors.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Matrix with res[i][j] = -manhattan_distance(a[i], b[j])
    """
    a = _convert_to_batch_tensor(a)
    b = _convert_to_batch_tensor(b)

    return -torch.cdist(a, b, p=1.0)


def pairwise_manhattan_sim(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor):
    """
    Computes the manhattan similarity (i.e., negative distance) between pairs of tensors.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Vector with res[i] = -manhattan_distance(a[i], b[i])
    """
    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)

    return -torch.sum(torch.abs(a - b), dim=-1)


def euclidean_sim(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor) -> Tensor:
    """
    Computes the euclidean similarity (i.e., negative distance) between two tensors.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Matrix with res[i][j] = -euclidean_distance(a[i], b[j])
    """
    a = _convert_to_batch_tensor(a)
    b = _convert_to_batch_tensor(b)

    return -torch.cdist(a, b, p=2.0)


def pairwise_euclidean_sim(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor):
    """
    Computes the euclidean distance (i.e., negative distance) between pairs of tensors.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Vector with res[i] = -euclidean_distance(a[i], b[i])
    """
    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)

    return -torch.sqrt(torch.sum((a - b) ** 2, dim=-1))


def pairwise_angle_sim(x: Tensor, y: Tensor) -> Tensor:
    """
    Computes the absolute normalized angle distance. See :class:`~sentence_transformers.losses.AnglELoss`
    or https://arxiv.org/abs/2309.12871v1 for more information.

    Args:
        x (Tensor): The first tensor.
        y (Tensor): The second tensor.

    Returns:
        Tensor: Vector with res[i] = angle_sim(a[i], b[i])
    """

    x = _convert_to_tensor(x)
    y = _convert_to_tensor(y)

    # modified from https://github.com/SeanLee97/AnglE/blob/main/angle_emb/angle.py
    # chunk both tensors to obtain complex components
    a, b = torch.chunk(x, 2, dim=1)
    c, d = torch.chunk(y, 2, dim=1)

    z = torch.sum(c**2 + d**2, dim=1, keepdim=True)
    re = (a * c + b * d) / z
    im = (b * c - a * d) / z

    dz = torch.sum(a**2 + b**2, dim=1, keepdim=True) ** 0.5
    dw = torch.sum(c**2 + d**2, dim=1, keepdim=True) ** 0.5
    re /= dz / dw
    im /= dz / dw

    norm_angle = torch.sum(torch.concat((re, im), dim=1), dim=1)
    return torch.abs(norm_angle)


def normalize_embeddings(embeddings: Tensor) -> Tensor:
    """
    Normalizes the embeddings matrix, so that each sentence embedding has unit length.

    Args:
        embeddings (Tensor): The input embeddings matrix.

    Returns:
        Tensor: The normalized embeddings matrix.
    """
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


@overload
def truncate_embeddings(embeddings: np.ndarray, truncate_dim: int | None) -> np.ndarray: ...


@overload
def truncate_embeddings(embeddings: torch.Tensor, truncate_dim: int | None) -> torch.Tensor: ...


def truncate_embeddings(embeddings: np.ndarray | torch.Tensor, truncate_dim: int | None) -> np.ndarray | torch.Tensor:
    """
    Truncates the embeddings matrix.

    Args:
        embeddings (Union[np.ndarray, torch.Tensor]): Embeddings to truncate.
        truncate_dim (Optional[int]): The dimension to truncate sentence embeddings to. `None` does no truncation.

    Example:
        >>> from sentence_transformers import SentenceTransformer
        >>> from sentence_transformers.util import truncate_embeddings
        >>> model = SentenceTransformer("tomaarsen/mpnet-base-nli-matryoshka")
        >>> embeddings = model.encode(["It's so nice outside!", "Today is a beautiful day.", "He drove to work earlier"])
        >>> embeddings.shape
        (3, 768)
        >>> model.similarity(embeddings, embeddings)
        tensor([[1.0000, 0.8100, 0.1426],
                [0.8100, 1.0000, 0.2121],
                [0.1426, 0.2121, 1.0000]])
        >>> truncated_embeddings = truncate_embeddings(embeddings, 128)
        >>> truncated_embeddings.shape
        >>> model.similarity(truncated_embeddings, truncated_embeddings)
        tensor([[1.0000, 0.8092, 0.1987],
                [0.8092, 1.0000, 0.2716],
                [0.1987, 0.2716, 1.0000]])

    Returns:
        Union[np.ndarray, torch.Tensor]: Truncated embeddings.
    """
    return embeddings[..., :truncate_dim]


def paraphrase_mining(
    model,
    sentences: list[str],
    show_progress_bar: bool = False,
    batch_size: int = 32,
    query_chunk_size: int = 5000,
    corpus_chunk_size: int = 100000,
    max_pairs: int = 500000,
    top_k: int = 100,
    score_function: Callable[[Tensor, Tensor], Tensor] = cos_sim,
) -> list[list[float | int]]:
    """
    Given a list of sentences / texts, this function performs paraphrase mining. It compares all sentences against all
    other sentences and returns a list with the pairs that have the highest cosine similarity score.

    Args:
        model (SentenceTransformer): SentenceTransformer model for embedding computation
        sentences (List[str]): A list of strings (texts or sentences)
        show_progress_bar (bool, optional): Plotting of a progress bar. Defaults to False.
        batch_size (int, optional): Number of texts that are encoded simultaneously by the model. Defaults to 32.
        query_chunk_size (int, optional): Search for most similar pairs for #query_chunk_size at the same time. Decrease, to lower memory footprint (increases run-time). Defaults to 5000.
        corpus_chunk_size (int, optional): Compare a sentence simultaneously against #corpus_chunk_size other sentences. Decrease, to lower memory footprint (increases run-time). Defaults to 100000.
        max_pairs (int, optional): Maximal number of text pairs returned. Defaults to 500000.
        top_k (int, optional): For each sentence, we retrieve up to top_k other sentences. Defaults to 100.
        score_function (Callable[[Tensor, Tensor], Tensor], optional): Function for computing scores. By default, cosine similarity. Defaults to cos_sim.

    Returns:
        List[List[Union[float, int]]]: Returns a list of triplets with the format [score, id1, id2]
    """

    # Compute embedding for the sentences
    embeddings = model.encode(
        sentences, show_progress_bar=show_progress_bar, batch_size=batch_size, convert_to_tensor=True
    )

    return paraphrase_mining_embeddings(
        embeddings,
        query_chunk_size=query_chunk_size,
        corpus_chunk_size=corpus_chunk_size,
        max_pairs=max_pairs,
        top_k=top_k,
        score_function=score_function,
    )


def paraphrase_mining_embeddings(
    embeddings: Tensor,
    query_chunk_size: int = 5000,
    corpus_chunk_size: int = 100000,
    max_pairs: int = 500000,
    top_k: int = 100,
    score_function: Callable[[Tensor, Tensor], Tensor] = cos_sim,
) -> list[list[float | int]]:
    """
    Given a list of sentences / texts, this function performs paraphrase mining. It compares all sentences against all
    other sentences and returns a list with the pairs that have the highest cosine similarity score.

    Args:
        embeddings (Tensor): A tensor with the embeddings
        query_chunk_size (int): Search for most similar pairs for #query_chunk_size at the same time. Decrease, to lower memory footprint (increases run-time).
        corpus_chunk_size (int): Compare a sentence simultaneously against #corpus_chunk_size other sentences. Decrease, to lower memory footprint (increases run-time).
        max_pairs (int): Maximal number of text pairs returned.
        top_k (int): For each sentence, we retrieve up to top_k other sentences
        score_function (Callable[[Tensor, Tensor], Tensor]): Function for computing scores. By default, cosine similarity.

    Returns:
        List[List[Union[float, int]]]: Returns a list of triplets with the format [score, id1, id2]
    """

    top_k += 1  # A sentence has the highest similarity to itself. Increase +1 as we are interest in distinct pairs

    # Mine for duplicates
    pairs = queue.PriorityQueue()
    min_score = -1
    num_added = 0

    for corpus_start_idx in range(0, len(embeddings), corpus_chunk_size):
        for query_start_idx in range(0, len(embeddings), query_chunk_size):
            scores = score_function(
                embeddings[query_start_idx : query_start_idx + query_chunk_size],
                embeddings[corpus_start_idx : corpus_start_idx + corpus_chunk_size],
            )

            scores_top_k_values, scores_top_k_idx = torch.topk(
                scores, min(top_k, len(scores[0])), dim=1, largest=True, sorted=False
            )
            scores_top_k_values = scores_top_k_values.cpu().tolist()
            scores_top_k_idx = scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(scores)):
                for top_k_idx, corpus_itr in enumerate(scores_top_k_idx[query_itr]):
                    i = query_start_idx + query_itr
                    j = corpus_start_idx + corpus_itr

                    if i != j and scores_top_k_values[query_itr][top_k_idx] > min_score:
                        pairs.put((scores_top_k_values[query_itr][top_k_idx], i, j))
                        num_added += 1

                        if num_added >= max_pairs:
                            entry = pairs.get()
                            min_score = entry[0]

    # Get the pairs
    added_pairs = set()  # Used for duplicate detection
    pairs_list = []
    while not pairs.empty():
        score, i, j = pairs.get()
        sorted_i, sorted_j = sorted([i, j])

        if sorted_i != sorted_j and (sorted_i, sorted_j) not in added_pairs:
            added_pairs.add((sorted_i, sorted_j))
            pairs_list.append([score, sorted_i, sorted_j])

    # Highest scores first
    pairs_list = sorted(pairs_list, key=lambda x: x[0], reverse=True)
    return pairs_list


def information_retrieval(*args, **kwargs) -> list[list[dict[str, int | float]]]:
    """This function is deprecated. Use semantic_search instead"""
    return semantic_search(*args, **kwargs)


def semantic_search(
    query_embeddings: Tensor,
    corpus_embeddings: Tensor,
    query_chunk_size: int = 100,
    corpus_chunk_size: int = 500000,
    top_k: int = 10,
    score_function: Callable[[Tensor, Tensor], Tensor] = cos_sim,
) -> list[list[dict[str, int | float]]]:
    """
    This function performs a cosine similarity search between a list of query embeddings  and a list of corpus embeddings.
    It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.

    Args:
        query_embeddings (:class:`~torch.Tensor`): A 2 dimensional tensor with the query embeddings.
        corpus_embeddings (:class:`~torch.Tensor`): A 2 dimensional tensor with the corpus embeddings.
        query_chunk_size (int, optional): Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory. Defaults to 100.
        corpus_chunk_size (int, optional): Scans the corpus 100k entries at a time. Increasing that value increases the speed, but requires more memory. Defaults to 500000.
        top_k (int, optional): Retrieve top k matching entries. Defaults to 10.
        score_function (Callable[[:class:`~torch.Tensor`, :class:`~torch.Tensor`], :class:`~torch.Tensor`], optional): Function for computing scores. By default, cosine similarity.

    Returns:
        List[List[Dict[str, Union[int, float]]]]: A list with one entry for each query. Each entry is a list of dictionaries with the keys 'corpus_id' and 'score', sorted by decreasing cosine similarity scores.
    """

    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)

    # Check that corpus and queries are on the same device
    if corpus_embeddings.device != query_embeddings.device:
        query_embeddings = query_embeddings.to(corpus_embeddings.device)

    queries_result_list = [[] for _ in range(len(query_embeddings))]

    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
        # Iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # Compute cosine similarities
            cos_scores = score_function(
                query_embeddings[query_start_idx : query_start_idx + query_chunk_size],
                corpus_embeddings[corpus_start_idx : corpus_start_idx + corpus_chunk_size],
            )

            # Get top-k scores
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores, min(top_k, len(cos_scores[0])), dim=1, largest=True, sorted=False
            )
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(cos_scores)):
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    if len(queries_result_list[query_id]) < top_k:
                        heapq.heappush(
                            queries_result_list[query_id], (score, corpus_id)
                        )  # heaqp tracks the quantity of the first element in the tuple
                    else:
                        heapq.heappushpop(queries_result_list[query_id], (score, corpus_id))

    # change the data format and sort
    for query_id in range(len(queries_result_list)):
        for doc_itr in range(len(queries_result_list[query_id])):
            score, corpus_id = queries_result_list[query_id][doc_itr]
            queries_result_list[query_id][doc_itr] = {"corpus_id": corpus_id, "score": score}
        queries_result_list[query_id] = sorted(queries_result_list[query_id], key=lambda x: x["score"], reverse=True)

    return queries_result_list


def mine_hard_negatives(
    dataset: Dataset,
    model: SentenceTransformer,
    anchor_column_name: str | None = None,
    positive_column_name: str | None = None,
    corpus: list[str] | None = None,
    cross_encoder: CrossEncoder | None = None,
    range_min: int = 0,
    range_max: int | None = None,
    max_score: float | None = None,
    min_score: float | None = None,
    margin: float | None = None,
    num_negatives: int = 3,
    sampling_strategy: Literal["random", "top"] = "top",
    include_positives: bool = False,
    output_format: Literal["triplet", "n-tuple", "labeled-pair", "labeled-list"] = "triplet",
    batch_size: int = 32,
    faiss_batch_size: int = 16384,
    use_faiss: bool = False,
    use_multi_process: list[str] | bool = False,
    verbose: bool = True,
    as_triplets: bool | None = None,
) -> Dataset:
    """
    Add hard negatives to a dataset of (anchor, positive) pairs to create (anchor, positive, negative) triplets or
    (anchor, positive, negative_1, ..., negative_n) tuples.

    Hard negative mining is a technique to improve the quality of a dataset by adding hard negatives, which are
    texts that may appear similar to the anchor, but are not. Using hard negatives can improve the performance of
    models trained on the dataset.

    This function uses a SentenceTransformer model to embed the sentences in the dataset, and then finds the closest
    matches to each anchor sentence in the dataset. It then samples negatives from the closest matches, optionally
    using a CrossEncoder model to rescore the candidates.

    You can influence the candidate negative selection in various ways:

    - **range_min**: Minimum rank of the closest matches to consider as negatives: useful to skip the most similar texts to
      avoid marking texts as negative that are actually positives.
    - **range_max**: Maximum rank of the closest matches to consider as negatives: useful to limit the number of candidates
      to sample negatives from. A lower value makes processing faster, but may result in less candidate negatives that
      satisfy the margin or max_score conditions.
    - **max_score**: Maximum score to consider as a negative: useful to skip candidates that are too similar to the anchor.
    - **min_score**: Minimum score to consider as a negative: useful to skip candidates that are too dissimilar to the anchor.
    - **margin**: Margin for hard negative mining: useful to skip candidates negatives whose similarity to the anchor is
      within a certain margin of the positive pair. A value of 0 can be used to enforce that the negative is always
      further away from the anchor than the positive.
    - **sampling_strategy**: Sampling strategy for negatives: "top" or "random". "top" will always sample the top n
      candidates as negatives, while "random" will sample n negatives randomly from the candidates that satisfy the
      margin or max_score conditions.

    Example:

        >>> from sentence_transformers.util import mine_hard_negatives
        >>> from sentence_transformers import SentenceTransformer
        >>> from datasets import load_dataset
        >>> # Load a Sentence Transformer model
        >>> model = SentenceTransformer("all-MiniLM-L6-v2")
        >>>
        >>> # Load a dataset to mine hard negatives from
        >>> dataset = load_dataset("sentence-transformers/natural-questions", split="train")
        >>> dataset
        Dataset({
            features: ['query', 'answer'],
            num_rows: 100231
        })
        >>> dataset = mine_hard_negatives(
        ...     dataset=dataset,
        ...     model=model,
        ...     range_min=10,
        ...     range_max=50,
        ...     max_score=0.8,
        ...     margin=0.1,
        ...     num_negatives=5,
        ...     sampling_strategy="random",
        ...     batch_size=128,
        ...     use_faiss=True,
        ... )
        Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 784/784 [00:43<00:00, 17.83it/s]
        Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 784/784 [00:07<00:00, 99.60it/s]
        Querying FAISS index: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 784/784 [00:00<00:00, 884.99it/s]
        Metric       Positive       Negative     Difference
        Count         100,231        431,255        431,255
        Mean           0.6866         0.4289         0.2804
        Median         0.7010         0.4193         0.2740
        Std            0.1125         0.0754         0.0999
        Min            0.0303         0.1720         0.1001
        25%            0.6221         0.3747         0.1991
        50%            0.7010         0.4193         0.2740
        75%            0.7667         0.4751         0.3530
        Max            0.9584         0.7743         0.7003
        Skipped 1289492 potential negatives (25.23%) due to the margin of 0.1.
        Skipped 39 potential negatives (0.00%) due to the maximum score of 0.8.
        Could not find enough negatives for 69900 samples (13.95%). Consider adjusting the range_max, range_min, margin and max_score parameters if you'd like to find more valid negatives.
        >>> # Note: The minimum similarity difference is 0.1001 due to our margin of 0.1
        >>> dataset
        Dataset({
            features: ['query', 'answer', 'negative'],
            num_rows: 431255
        })
        >>> dataset[0]
        {
            'query': 'when did richmond last play in a preliminary final',
            'answer': "Richmond Football Club Richmond began 2017 with 5 straight wins, a feat it had not achieved since 1995. A series of close losses hampered the Tigers throughout the middle of the season, including a 5-point loss to the Western Bulldogs, 2-point loss to Fremantle, and a 3-point loss to the Giants. Richmond ended the season strongly with convincing victories over Fremantle and St Kilda in the final two rounds, elevating the club to 3rd on the ladder. Richmond's first final of the season against the Cats at the MCG attracted a record qualifying final crowd of 95,028; the Tigers won by 51 points. Having advanced to the first preliminary finals for the first time since 2001, Richmond defeated Greater Western Sydney by 36 points in front of a crowd of 94,258 to progress to the Grand Final against Adelaide, their first Grand Final appearance since 1982. The attendance was 100,021, the largest crowd to a grand final since 1986. The Crows led at quarter time and led by as many as 13, but the Tigers took over the game as it progressed and scored seven straight goals at one point. They eventually would win by 48 points – 16.12 (108) to Adelaide's 8.12 (60) – to end their 37-year flag drought.[22] Dustin Martin also became the first player to win a Premiership medal, the Brownlow Medal and the Norm Smith Medal in the same season, while Damien Hardwick was named AFL Coaches Association Coach of the Year. Richmond's jump from 13th to premiers also marked the biggest jump from one AFL season to the next.",
            'negative': "2018 NRL Grand Final The 2018 NRL Grand Final was the conclusive and premiership-deciding game of the 2018 National Rugby League season and was played on Sunday September 30 at Sydney's ANZ Stadium.[1] The match was contested between minor premiers the Sydney Roosters and defending premiers the Melbourne Storm. In front of a crowd of 82,688, Sydney won the match 21â€“6 to claim their 14th premiership title and their first since 2013. Roosters five-eighth Luke Keary was awarded the Clive Churchill Medal as the game's official man of the match."
        }
        >>> dataset.push_to_hub("natural-questions-hard-negatives", "triplet-all")

    Args:
        dataset (Dataset): A dataset containing (anchor, positive) pairs.
        model (SentenceTransformer): A SentenceTransformer model to use for embedding the sentences.
        anchor_column_name (str, optional): The column name in `dataset` that contains the anchor/query. Defaults to None, in which case the first column in `dataset` will be used.
        positive_column_name (str, optional): The column name in `dataset` that contains the positive candidates. Defaults to None, in which case the second column in `dataset` will be used.
        corpus (List[str], optional): A list containing documents as strings that will be used as candidate negatives
            in addition to the second column in `dataset`. Defaults to None, in which case the second column in
            `dataset` will exclusively be used as the negative candidate corpus.
        cross_encoder (CrossEncoder, optional): A CrossEncoder model to use for rescoring the candidates. Defaults to None.
        range_min (int): Minimum rank of the closest matches to consider as negatives. Defaults to 0.
        range_max (int, optional): Maximum rank of the closest matches to consider as negatives. Defaults to None.
        max_score (float, optional): Maximum score to consider as a negative. Defaults to None.
        min_score (float, optional): Minimum score to consider as a negative. Defaults to None.
        margin (float, optional): Margin for hard negative mining. Defaults to None.
        num_negatives (int): Number of negatives to sample. Defaults to 3.
        sampling_strategy (Literal["random", "top"]): Sampling strategy for negatives: "top" or "random". Defaults to "top".
        include_positives (bool): Whether to include the positives in the negative candidates.
            Setting this to True is primarily useful for creating Reranking evaluation datasets for CrossEncoder models,
            where it can be useful to get a full ranking (including the positives) from a first-stage retrieval model.
            Defaults to False.
        output_format (Literal["triplet", "n-tuple", "labeled-pair", "labeled-list"]): Output format for the `datasets.Dataset`. Options are:

            - "triplet": (anchor, positive, negative) triplets, i.e. 3 columns. Useful for e.g. :class:`~sentence_transformers.cross_encoder.losses.CachedMultipleNegativesRankingLoss`.
            - "n-tuple": (anchor, positive, negative_1, ..., negative_n) tuples, i.e. 2 + num_negatives columns. Useful for e.g. :class:`~sentence_transformers.cross_encoder.losses.CachedMultipleNegativesRankingLoss`.
            - "labeled-pair": (anchor, passage, label) text tuples with a label of 0 for negative and 1 for positive, i.e. 3 columns. Useful for e.g. :class:`~sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss`.
            - "labeled-list": (anchor, [doc1, doc2, ..., docN], [label1, label2, ..., labelN]) triplets with labels of 0 for negative and 1 for positive, i.e. 3 columns. Useful for e.g. :class:`~sentence_transformers.cross_encoder.losses.LambdaLoss`.

            Defaults to "triplet".
        batch_size (int): Batch size for encoding the dataset. Defaults to 32.
        faiss_batch_size (int): Batch size for FAISS top-k search. Defaults to 16384.
        use_faiss (bool): Whether to use FAISS for similarity search. May be recommended for large datasets. Defaults to False.
        use_multi_process (bool | List[str], optional): Whether to use multi-GPU/CPU processing. If True, uses all GPUs if CUDA
            is available, and 4 CPU processes if it's not available. You can also pass a list of PyTorch devices like
            ["cuda:0", "cuda:1", ...] or ["cpu", "cpu", "cpu", "cpu"].
        verbose (bool): Whether to print statistics and logging. Defaults to True.
        as_triplets (bool, optional): Deprecated. Use `output_format` instead. Defaults to None.

    Returns:
        Dataset: A dataset containing (anchor, positive, negative) triplets, (anchor, passage, label) text tuples with
        a label, or (anchor, positive, negative_1, ..., negative_n) tuples.
    """
    if not is_datasets_available():
        raise ImportError("Please install `datasets` to use this function: `pip install datasets`.")

    from datasets import Dataset

    # If a dataset has duplicate queries, assume that all duplicates are positive pairs.
    columns = dataset.column_names

    if not anchor_column_name or anchor_column_name not in columns:
        anchor_column_name = columns[0]

    if not positive_column_name or positive_column_name not in columns:
        positive_column_name = columns[1]

    if not anchor_column_name and not positive_column_name and len(columns) != 2:
        raise ValueError("Dataset must contain exactly two columns.")

    if as_triplets is not None:
        output_format = "triplet" if as_triplets else "n-tuple"
        logger.warning(
            "The `as_triplets` parameter is deprecated. Use the `output_format` parameter instead. "
            f"Setting `output_format` to `{output_format}`."
        )

    if include_positives:
        if (
            range_min != 0
            or range_max is not None
            or max_score is not None
            or margin is not None
            or sampling_strategy != "top"
        ):
            logger.warning(
                "When using `include_positives=True`, updating `range_min`, `range_max`, `max_score`, `margin`, or "
                "`sampling_strategy` from the default values may still discard the positive values."
            )
        if output_format != "n-tuple":
            logger.warning(
                'When using `include_positives=True`, `output_format` will be set to `"n-tuple"` to ensure that the ranking order is preserved.'
            )
            output_format = "n-tuple"

    # To avoid re-embedding the same query multiple times, we keep a counter of the number of positives per query
    positives_per_query = list(
        dataset.to_pandas().groupby(anchor_column_name).count().to_dict()[positive_column_name].values()
    )
    max_positives = max(positives_per_query)

    if range_max is None:
        if margin is not None or max_score is not None:
            # max_positives + 10 * num_negatives negatives because some might be skipped, and range_min skipped
            range_max = range_min + (num_negatives * 10) + max_positives
        else:
            # max_positives, num_negatives negatives, and range_min skipped
            range_max = range_min + num_negatives + max_positives
        if range_max > 2048 and use_faiss:
            # FAISS on GPU can only retrieve up to 2048 documents per query
            range_max = 2048
            if verbose:
                print("Using FAISS, we can only retrieve up to 2048 documents per query. Setting range_max to 2048.")
        if verbose:
            print(f"Setting range_max to {range_max} based on the provided parameters.")

    log_counters = {}
    queries = dataset[anchor_column_name]
    positives = dataset[positive_column_name]
    separate_corpus = corpus is not None
    if not separate_corpus:
        corpus = positives

    # Deduplicate the corpus
    # make sure all the positives are also in the corpus and de-duplicate it.
    corpus = list(set(corpus) | set(positives))

    # corpus_idx maps the corpus text into its position in the corpus
    # This position does not necessarily matches the original corpus, as it was de-duplicated.
    corpus_idx = {text: idx for idx, text in enumerate(corpus)}

    # Deduplicate the queries, but keep the original one for later reference.
    all_queries = queries.copy()
    queries = list(set(queries))
    queries_idx = {query: idx for idx, query in enumerate(queries)}
    n_queries = len(queries)
    batch_idx = torch.arange(n_queries).unsqueeze(-1)

    device = model.device

    if n_queries != len(all_queries) and verbose:
        print(f"Found {n_queries} unique queries out of {len(all_queries)} total queries.")

    if max_positives > 1:
        avg_positives_per_query = np.mean(positives_per_query)
        print(f"Found an average of {avg_positives_per_query:.3f} positives per query.")

    # Embed the corpus and the queries
    if use_multi_process:
        pool = model.start_multi_process_pool(
            target_devices=None if isinstance(use_multi_process, bool) else use_multi_process
        )
        corpus_embeddings = model.encode_multi_process(
            corpus, pool, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
        )
        query_embeddings = model.encode_multi_process(
            queries, pool, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
        )
        model.stop_multi_process_pool(pool)
    else:
        corpus_embeddings = model.encode(
            corpus, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
        )
        query_embeddings = model.encode(
            queries,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    if use_faiss:
        import faiss

        index = faiss.IndexFlatIP(model.get_sentence_embedding_dimension())
        # Move the index to the GPU if available
        try:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            index: faiss.IndexFlatIP = faiss.index_cpu_to_all_gpus(index, co=co)
        except Exception:
            pass

        index.add(corpus_embeddings)

        scores_list = []
        indices_list = []
        # Iterate over query embeddings in batches so we can track the progress
        for i in trange(0, len(query_embeddings), faiss_batch_size, desc="Querying FAISS index"):
            query_chunk = query_embeddings[i : i + faiss_batch_size]
            scores, indices = index.search(query_chunk, k=range_max + 1)
            scores_list.append(scores)
            indices_list.append(indices)
        scores = torch.from_numpy(np.concatenate(scores_list, axis=0)).to(device)
        indices = torch.from_numpy(np.concatenate(indices_list, axis=0)).to(device)

    else:
        # Compute the similarity scores between the queries and the corpus
        scores = model.similarity(query_embeddings, corpus_embeddings).to(device)

        # Keep only the range_max + max_positives highest scores. We offset by 1 to potentially include the positive pair
        scores, indices = torch.topk(scores, k=range_max + max_positives, dim=1)

    # As we may have duplicated queries (i.e., a single query with multiple positives),
    # We keep track, for each unique query, of where their positives are in the list of positives (positive_indices).
    # Note that as queries may have differing numbers of positives, we cannot guarantee that this is a fixed-length matrix.
    positive_indices = [[] for _ in range(n_queries)]

    for query, positive in zip(all_queries, positives):
        query_idx = queries_idx[query]
        positive_indices[query_idx].append(corpus_idx[positive])

    n_positives = [len(p) for p in positive_indices]

    # re-sort the positives and all_queries according to the deduplicated queries
    positives = []
    all_queries = []
    for idx in range(n_queries):
        positives.extend([corpus[doc_idx] for doc_idx in positive_indices[idx]])
        all_queries.extend([queries[idx]] * n_positives[idx])

    positive_indices = [torch.tensor(p, device=device) for p in positive_indices]

    # Compute the positive scores
    query_embeddings = query_embeddings[[idx for idx in range(n_queries) for _ in range(n_positives[idx])]]
    positive_embeddings = corpus_embeddings[torch.cat(positive_indices).tolist()]
    positive_scores = model.similarity_pairwise(query_embeddings, positive_embeddings).to(device)

    del query_embeddings
    del positive_embeddings
    del corpus_embeddings

    # Rescore with cross_encoder
    if cross_encoder is not None and (margin is not None or max_score is not None):
        for idx, candidate_idx in tqdm(enumerate(indices), desc="Rescoring with CrossEncoder", total=len(indices)):
            query = queries[idx]
            candidate_passages = [corpus[_idx] for _idx in candidate_idx]
            pred_scores = cross_encoder.predict(
                list(zip([query] * (range_max + 1), candidate_passages)),
                batch_size=batch_size,
                convert_to_tensor=True,
            )
            scores[idx] = pred_scores
        positive_scores = cross_encoder.predict(
            list(zip(all_queries, positives)),
            batch_size=batch_size,
            convert_to_tensor=True,
        )

    if not include_positives:
        # for each query, create a mask that is True for the positives and False for the negatives in the indices
        positive_mask = torch.stack(
            [torch.isin(indices[q_idx], positive_indices[q_idx]) for q_idx in range(n_queries)]
        )

        # Scores is a [num_queries, range_max] tensor, where we set the values to -inf to disqualify the corresponding
        # positive candidates
        scores[positive_mask] = -float("inf")

    num_candidates = scores.numel()

    # Remove based on margin
    if margin is not None:
        # If we have a margin, we will remove candidates that are too close to the positive pair
        # If there are multiple positives, we need to define which one to use for the margin
        # To be on the safe side, we will use the _minimum_ positive score (i.e., harder positive) for the margin
        max_positive_scores = torch.empty(n_queries, device=positive_scores.device, dtype=positive_scores.dtype)
        start_idx = 0
        for q_idx in range(n_queries):
            max_positive_scores[q_idx] = torch.min(positive_scores[start_idx : start_idx + n_positives[q_idx]])
            start_idx += n_positives[q_idx - 1]

        removed_indices = scores + margin > max_positive_scores.repeat(scores.size(1), 1).T
        scores[removed_indices] = -float("inf")

        num_skipped = removed_indices.sum().item()
        if num_skipped:
            log_counters["margin"] = {
                "skipped": num_skipped,
                "ratio": num_skipped / num_candidates,
            }
            num_candidates -= num_skipped

    # Remove based on max_score
    if max_score is not None:
        removed_indices = scores > max_score
        scores[removed_indices] = -float("inf")

        num_skipped = removed_indices.sum().item()
        if num_skipped:
            log_counters["max_score"] = {
                "skipped": num_skipped,
                "ratio": num_skipped / num_candidates,
            }

    # Remove based on min_score
    if min_score is not None:
        removed_indices = scores < min_score
        scores[removed_indices] = -float("inf")

        num_skipped = removed_indices.sum().item()
        if num_skipped:
            log_counters["min_score"] = {
                "skipped": num_skipped,
                "ratio": num_skipped / num_candidates,
            }

    # Grab the top negative candidates and remove the first range_min candidates
    negative_scores, local_indices = torch.topk(scores, k=range_max, dim=1)
    indices = indices[batch_idx, local_indices]

    if range_min:
        indices = indices[:, range_min:]
        negative_scores = negative_scores[:, range_min:]

    # Either grab the top negatives or sample randomly
    if sampling_strategy == "top":
        indices = indices[:, :num_negatives]
        negative_scores = negative_scores[:, :num_negatives]

    elif sampling_strategy == "random":
        # Prevent sampling -inf values if possible
        num_options = indices.size(1) - negative_scores.isinf().sum(1)
        num_options = num_options.clamp(min=num_negatives)
        # Randomly sample negatives from each row
        sampled_idx = [random.sample(range(options), k=num_negatives) for options in num_options]
        indices = indices[batch_idx, sampled_idx]
        negative_scores = negative_scores[batch_idx, sampled_idx]
        # Resort the indices and scores
        negative_scores, local_indices = negative_scores.sort(dim=1, descending=True)
        indices = indices[batch_idx, local_indices]

    # repeat indices and negative_scores by the number of positives of each query
    indices = torch.cat([indices[idx].repeat(n_positives[idx], 1) for idx in range(n_queries)])
    negative_scores = torch.cat([negative_scores[idx].repeat(n_positives[idx], 1) for idx in range(n_queries)])

    if output_format == "triplet":
        # If calling as triples and there are multiple positives per query, we will explode the dataset into triplets.
        indices_to_keep = negative_scores != -float("inf")
        anchor_indices = torch.empty_like(indices)
        pos_indices = torch.empty_like(indices)

        indices = indices[indices_to_keep]
        negative_scores = negative_scores[indices_to_keep]

        # the anchor_indices matrix is shaped [n_total_queries, n_negatives]
        start_idx = 0
        for q_idx in range(n_queries):
            anchor_indices[start_idx : start_idx + n_positives[q_idx]] = torch.tensor(q_idx).repeat(
                n_positives[q_idx], num_negatives
            )
            pos_indices[start_idx : start_idx + n_positives[q_idx]] = (
                positive_indices[q_idx].repeat(num_negatives, 1).T
            )
            start_idx += n_positives[q_idx]

        anchor_indices = anchor_indices[indices_to_keep]
        positive_indices = pos_indices[indices_to_keep]

        dataset_data = {
            anchor_column_name: [],
            positive_column_name: [],
            "negative": [],
        }

        for anchor_idx, positive_idx, negative_idx in zip(anchor_indices, positive_indices, indices):
            dataset_data[anchor_column_name].append(queries[anchor_idx])
            dataset_data[positive_column_name].append(corpus[positive_idx])
            dataset_data["negative"].append(corpus[negative_idx])
        difference_scores = positive_scores.repeat(num_negatives, 1).T[indices_to_keep] - negative_scores

    elif output_format == "labeled-pair":
        indices_to_keep = negative_scores != -float("inf")

        dataset_data = {
            anchor_column_name: [],
            positive_column_name: [],  # Note, this is not strictly positives
            "label": [],
        }

        for query_idx in range(n_queries):
            for positive_idx in positive_indices[query_idx]:
                dataset_data[anchor_column_name].append(queries[query_idx])
                dataset_data[positive_column_name].append(corpus[positive_idx])
                dataset_data["label"].append(1)
            for negative_idx, negative_score in zip(indices[query_idx], negative_scores[query_idx]):
                if negative_score == -float("inf"):
                    continue
                dataset_data[anchor_column_name].append(queries[query_idx])
                dataset_data[positive_column_name].append(corpus[negative_idx])
                dataset_data["label"].append(0)

        negative_scores = negative_scores[indices_to_keep]
        difference_scores = positive_scores.repeat(num_negatives, 1).T[indices_to_keep] - negative_scores

    elif output_format == "n-tuple":
        # Keep only indices where num_negative negatives were found
        indices_to_keep = (negative_scores != -float("inf")).all(dim=1)
        negative_scores = negative_scores[indices_to_keep]
        indices = indices[indices_to_keep]

        dataset_data = {
            anchor_column_name: [all_queries[idx] for idx, keep in enumerate(indices_to_keep) if keep],
            positive_column_name: [positives[idx] for idx, keep in enumerate(indices_to_keep) if keep],
            **{
                f"negative_{i}": [corpus[neg_idx] for neg_idx in neg_indices]
                for i, neg_indices in enumerate(indices.T, start=1)
            },
        }
        negative_scores = negative_scores.flatten()
        difference_scores = positive_scores.repeat(num_negatives, 1).T[indices_to_keep].flatten() - negative_scores

    elif output_format == "labeled-list":
        indices_to_keep = negative_scores != -float("inf")

        dataset_data = {
            anchor_column_name: [all_queries[idx] for idx, keep_row in enumerate(indices_to_keep) if keep_row.any()],
            positive_column_name: [
                [positives[idx]] + [corpus[index] for keep, index in zip(keep_row, indices_row) if keep]
                for idx, (keep_row, indices_row) in enumerate(zip(indices_to_keep, indices))
                if keep_row.any()
            ],
            "labels": [[1] + [0] * sum(keep_row) for keep_row in indices_to_keep if keep_row.any()],
        }
        negative_scores = negative_scores[indices_to_keep]
        difference_scores = positive_scores.repeat(num_negatives, 1).T[indices_to_keep] - negative_scores

    if len(dataset_data) == 0:
        raise ValueError("No triplets could be generated. Please check the parameters and dataset.")
    output_dataset = Dataset.from_dict(dataset_data)

    # Report some statistics
    if verbose:
        row_format = "{:<6} {:>14} {:>14} {:>14}"
        formatter = lambda value: (f"{value.item():.4f}" if isinstance(value, torch.Tensor) else f"{value:,}")
        print(row_format.format("Metric", "Positive", "Negative", "Difference"))
        print(
            row_format.format(
                "Count",
                formatter(len(positive_scores)),
                formatter(len(negative_scores)),
                "",
            )
        )
        for metric, function in [
            ("mean", torch.mean),
            ("median", torch.median),
            ("std", torch.std),
            ("min", lambda scores: torch.min(scores) if scores.numel() > 0 else float("inf")),
            ("25%", lambda scores: torch.quantile(scores.float(), q=0.25) if scores.numel() > 0 else float("inf")),
            ("50%", lambda scores: torch.quantile(scores.float(), q=0.5) if scores.numel() > 0 else float("inf")),
            ("75%", lambda scores: torch.quantile(scores.float(), q=0.75) if scores.numel() > 0 else float("inf")),
            ("max", lambda scores: torch.max(scores) if scores.numel() > 0 else float("-inf")),
        ]:
            print(
                row_format.format(
                    metric.capitalize(),
                    formatter(function(positive_scores)),
                    formatter(function(negative_scores)),
                    formatter(function(difference_scores)),
                )
            )

        if "margin" in log_counters:
            print(
                f"Skipped {log_counters['margin']['skipped']} potential negatives ({log_counters['margin']['ratio']:.2%}) due to the margin of {margin}."
            )
        if "max_score" in log_counters:
            print(
                f"Skipped {log_counters['max_score']['skipped']} potential negatives ({log_counters['max_score']['ratio']:.2%}) due to the maximum score of {max_score}."
            )
        if "min_score" in log_counters:
            print(
                f"Skipped {log_counters['min_score']['skipped']} potential negatives ({log_counters['min_score']['ratio']:.2%}) due to the minimum score of {min_score}."
            )

        missing_negatives = (num_negatives * len(dataset)) - len(negative_scores)
        if missing_negatives > 0:
            solutions = ["range_max"]
            if range_min > 0:
                solutions.append("range_min")
            if margin is not None:
                solutions.append("margin")
            if max_score is not None:
                solutions.append("max_score")
            considerations = ", ".join(solutions[:-1])
            if len(solutions) > 1:
                considerations += " and " + solutions[-1]
            missing_negatives_ratio = missing_negatives / (num_negatives * len(dataset))
            print(
                f"Could not find enough negatives for {missing_negatives} samples ({missing_negatives_ratio:.2%})."
                f" Consider adjusting the {considerations} parameter{'s' if len(solutions) > 1 else ''} if you'd like to find more valid negatives."
            )

    return output_dataset


def http_get(url: str, path: str) -> None:
    """
    Downloads a URL to a given path on disk.

    Args:
        url (str): The URL to download.
        path (str): The path to save the downloaded file.

    Raises:
        requests.HTTPError: If the HTTP request returns a non-200 status code.

    Returns:
        None
    """
    if os.path.dirname(path) != "":
        os.makedirs(os.path.dirname(path), exist_ok=True)

    req = requests.get(url, stream=True)
    if req.status_code != 200:
        print(f"Exception when trying to download {url}. Response {req.status_code}", file=sys.stderr)
        req.raise_for_status()
        return

    download_filepath = path + "_part"
    with open(download_filepath, "wb") as file_binary:
        content_length = req.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                file_binary.write(chunk)

    os.rename(download_filepath, path)
    progress.close()


def batch_to_device(batch: dict[str, Any], target_device: device) -> dict[str, Any]:
    """
    Send a PyTorch batch (i.e., a dictionary of string keys to Tensors) to a device (e.g. "cpu", "cuda", "mps").

    Args:
        batch (Dict[str, Tensor]): The batch to send to the device.
        target_device (torch.device): The target device (e.g. "cpu", "cuda", "mps").

    Returns:
        Dict[str, Tensor]: The batch with tensors sent to the target device.
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


def fullname(o) -> str:
    """
    Gives a full name (package_name.class_name) for a class / object in Python. Will
    be used to load the correct classes from JSON files

    Args:
        o: The object for which to get the full name.

    Returns:
        str: The full name of the object.

    Example:
        >>> from sentence_transformers.losses import MultipleNegativesRankingLoss
        >>> from sentence_transformers import SentenceTransformer
        >>> from sentence_transformers.util import fullname
        >>> model = SentenceTransformer('all-MiniLM-L6-v2')
        >>> loss = MultipleNegativesRankingLoss(model)
        >>> fullname(loss)
        'sentence_transformers.losses.MultipleNegativesRankingLoss.MultipleNegativesRankingLoss'
    """

    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__  # Avoid reporting __builtin__
    else:
        return module + "." + o.__class__.__name__


def import_from_string(dotted_path: str) -> type:
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.

    Args:
        dotted_path (str): The dotted module path.

    Returns:
        Any: The attribute/class designated by the last name in the path.

    Raises:
        ImportError: If the import failed.

    Example:
        >>> import_from_string('sentence_transformers.losses.MultipleNegativesRankingLoss')
        <class 'sentence_transformers.losses.MultipleNegativesRankingLoss.MultipleNegativesRankingLoss'>
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError:
        msg = f"{dotted_path} doesn't look like a module path"
        raise ImportError(msg)

    try:
        module = importlib.import_module(dotted_path)
    except Exception:
        module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = f'Module "{module_path}" does not define a "{class_name}" attribute/class'
        raise ImportError(msg)


def community_detection(
    embeddings: torch.Tensor | np.ndarray,
    threshold: float = 0.75,
    min_community_size: int = 10,
    batch_size: int = 1024,
    show_progress_bar: bool = False,
) -> list[list[int]]:
    """
    Function for Fast Community Detection.

    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
    Returns only communities that are larger than min_community_size. The communities are returned
    in decreasing order. The first element in each list is the central point in the community.

    Args:
        embeddings (torch.Tensor or numpy.ndarray): The input embeddings.
        threshold (float): The threshold for determining if two embeddings are close. Defaults to 0.75.
        min_community_size (int): The minimum size of a community to be considered. Defaults to 10.
        batch_size (int): The batch size for computing cosine similarity scores. Defaults to 1024.
        show_progress_bar (bool): Whether to show a progress bar during computation. Defaults to False.

    Returns:
        List[List[int]]: A list of communities, where each community is represented as a list of indices.
    """
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)

    threshold = torch.tensor(threshold, device=embeddings.device)
    embeddings = normalize_embeddings(embeddings)

    extracted_communities = []

    # Maximum size for community
    min_community_size = min(min_community_size, len(embeddings))
    sort_max_size = min(max(2 * min_community_size, 50), len(embeddings))

    for start_idx in tqdm(
        range(0, len(embeddings), batch_size), desc="Finding clusters", disable=not show_progress_bar
    ):
        # Compute cosine similarity scores
        cos_scores = embeddings[start_idx : start_idx + batch_size] @ embeddings.T

        # Use a torch-heavy approach if the embeddings are on CUDA, otherwise a loop-heavy one
        if embeddings.device.type in ["cuda", "npu"]:
            # Threshold the cos scores and determine how many close embeddings exist per embedding
            threshold_mask = cos_scores >= threshold
            row_wise_count = threshold_mask.sum(1)

            # Only consider embeddings with enough close other embeddings
            large_enough_mask = row_wise_count >= min_community_size
            if not large_enough_mask.any():
                continue

            row_wise_count = row_wise_count[large_enough_mask]
            cos_scores = cos_scores[large_enough_mask]

            # The max is the largest potential community, so we use that in topk
            k = row_wise_count.max()
            _, top_k_indices = cos_scores.topk(k=k, largest=True)

            # Use the row-wise count to slice the indices
            for count, indices in zip(row_wise_count, top_k_indices):
                extracted_communities.append(indices[:count].tolist())
        else:
            # Minimum size for a community
            top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

            # Filter for rows >= min_threshold
            for i in range(len(top_k_values)):
                if top_k_values[i][-1] >= threshold:
                    # Only check top k most similar entries
                    top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                    # Check if we need to increase sort_max_size
                    while top_val_large[-1] > threshold and sort_max_size < len(embeddings):
                        sort_max_size = min(2 * sort_max_size, len(embeddings))
                        top_val_large, top_idx_large = cos_scores[i].topk(k=sort_max_size, largest=True)

                    extracted_communities.append(top_idx_large[top_val_large >= threshold].tolist())

    # Largest cluster first
    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for cluster_id, community in enumerate(extracted_communities):
        non_overlapped_community = []
        for idx in community:
            if idx not in extracted_ids:
                non_overlapped_community.append(idx)

        if len(non_overlapped_community) >= min_community_size:
            unique_communities.append(non_overlapped_community)
            extracted_ids.update(non_overlapped_community)

    unique_communities = sorted(unique_communities, key=lambda x: len(x), reverse=True)

    return unique_communities


##################
#
######################


class disabled_tqdm(tqdm):
    """
    Class to override `disable` argument in case progress bars are globally disabled.

    Taken from https://github.com/tqdm/tqdm/issues/619#issuecomment-619639324.
    """

    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)

    def __delattr__(self, attr: str) -> None:
        """Fix for https://github.com/huggingface/huggingface_hub/issues/1603"""
        try:
            super().__delattr__(attr)
        except AttributeError:
            if attr != "_lock":
                raise


@contextmanager
def disable_logging(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.

    Args:
        highest_level: the maximum logging level allowed.
    """

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)


def is_sentence_transformer_model(
    model_name_or_path: str,
    token: bool | str | None = None,
    cache_folder: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> bool:
    """
    Checks if the given model name or path corresponds to a SentenceTransformer model.

    Args:
        model_name_or_path (str): The name or path of the model.
        token (Optional[Union[bool, str]]): The token to be used for authentication. Defaults to None.
        cache_folder (Optional[str]): The folder to cache the model files. Defaults to None.
        revision (Optional[str]): The revision of the model. Defaults to None.
        local_files_only (bool): Whether to only use local files for the model. Defaults to False.

    Returns:
        bool: True if the model is a SentenceTransformer model, False otherwise.
    """
    return bool(
        load_file_path(
            model_name_or_path,
            "modules.json",
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
    )


def load_file_path(
    model_name_or_path: str,
    filename: str,
    token: bool | str | None = None,
    cache_folder: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> str | None:
    """
    Loads a file from a local or remote location.

    Args:
        model_name_or_path (str): The model name or path.
        filename (str): The name of the file to load.
        token (Optional[Union[bool, str]]): The token to access the remote file (if applicable).
        cache_folder (Optional[str]): The folder to cache the downloaded file (if applicable).
        revision (Optional[str], optional): The revision of the file (if applicable). Defaults to None.
        local_files_only (bool, optional): Whether to only consider local files. Defaults to False.

    Returns:
        Optional[str]: The path to the loaded file, or None if the file could not be found or loaded.
    """
    # If file is local
    file_path = os.path.join(model_name_or_path, filename)
    if os.path.exists(file_path):
        return file_path

    # If file is remote
    try:
        return hf_hub_download(
            model_name_or_path,
            filename=filename,
            revision=revision,
            library_name="sentence-transformers",
            token=token,
            cache_dir=cache_folder,
            local_files_only=local_files_only,
        )
    except Exception:
        return None


def load_dir_path(
    model_name_or_path: str,
    directory: str,
    token: bool | str | None = None,
    cache_folder: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> str | None:
    """
    Loads the directory path for a given model name or path.

    Args:
        model_name_or_path (str): The name or path of the model.
        directory (str): The directory to load.
        token (Optional[Union[bool, str]]): The token for authentication.
        cache_folder (Optional[str]): The folder to cache the downloaded files.
        revision (Optional[str], optional): The revision of the model. Defaults to None.
        local_files_only (bool, optional): Whether to only use local files. Defaults to False.

    Returns:
        Optional[str]: The directory path if it exists, otherwise None.
    """
    # If file is local
    dir_path = os.path.join(model_name_or_path, directory)
    if os.path.exists(dir_path):
        return dir_path

    download_kwargs = {
        "repo_id": model_name_or_path,
        "revision": revision,
        "allow_patterns": f"{directory}/**" if directory not in ["", "."] else None,
        "library_name": "sentence-transformers",
        "token": token,
        "cache_dir": cache_folder,
        "local_files_only": local_files_only,
        "tqdm_class": disabled_tqdm,
    }
    # Try to download from the remote
    try:
        repo_path = snapshot_download(**download_kwargs)
    except Exception:
        # Otherwise, try local (i.e. cache) only
        download_kwargs["local_files_only"] = True
        repo_path = snapshot_download(**download_kwargs)
    return os.path.join(repo_path, directory)


def save_to_hub_args_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # If repo_id not already set, use repo_name
        repo_name = kwargs.pop("repo_name", None)
        if repo_name and "repo_id" not in kwargs:
            logger.warning(
                "Providing a `repo_name` keyword argument to `save_to_hub` is deprecated, please use `repo_id` instead."
            )
            kwargs["repo_id"] = repo_name

        # If positional args are used, adjust for the new "token" keyword argument
        if len(args) >= 2:
            args = (*args[:2], None, *args[2:])

        return func(self, *args, **kwargs)

    return wrapper


def get_device_name() -> Literal["mps", "cuda", "npu", "hpu", "cpu"]:
    """
    Returns the name of the device where this module is running on.

    It's a simple implementation that doesn't cover cases when more powerful GPUs are available and
    not a primary device ('cuda:0') or MPS device is available, but not configured properly.

    Returns:
        str: Device name, like 'cuda' or 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    elif is_torch_npu_available():
        return "npu"
    elif importlib.util.find_spec("habana_frameworks") is not None:
        import habana_frameworks.torch.hpu as hthpu

        if hthpu.is_available():
            return "hpu"
    return "cpu"


def check_package_availability(package_name: str, owner: str) -> bool:
    """
    Checks if a package is available from the correct owner.
    """
    try:
        meta = metadata(package_name)
        return meta["Name"] == package_name and owner in meta["Home-page"]
    except PackageNotFoundError:
        return False


def is_accelerate_available() -> bool:
    """
    Returns True if the Huggingface accelerate library is available.
    """
    return check_package_availability("accelerate", "huggingface")


def is_datasets_available() -> bool:
    """
    Returns True if the Huggingface datasets library is available.
    """
    return check_package_availability("datasets", "huggingface")


def is_training_available() -> bool:
    """
    Returns True if we have the required dependencies for training Sentence
    Transformers models, i.e. Huggingface datasets and Huggingface accelerate.
    """
    return is_accelerate_available() and is_datasets_available()


@contextmanager
def disable_datasets_caching():
    """
    A context manager that will disable caching in the datasets library.
    """
    from datasets import disable_caching, enable_caching, is_caching_enabled

    is_originally_enabled = is_caching_enabled()

    try:
        if is_originally_enabled:
            disable_caching()
        yield
    finally:
        if is_originally_enabled:
            enable_caching()
