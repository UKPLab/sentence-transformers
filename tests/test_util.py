from __future__ import annotations

import numpy as np
import sklearn
import torch

from sentence_transformers import SentenceTransformer, util


def test_normalize_embeddings() -> None:
    """Tests the correct computation of util.normalize_embeddings"""
    embedding_size = 100
    a = torch.tensor(np.random.randn(50, embedding_size))
    a_norm = util.normalize_embeddings(a)

    for embedding in a_norm:
        assert len(embedding) == embedding_size
        emb_norm = torch.norm(embedding)
        assert abs(emb_norm.item() - 1) < 0.0001


def test_pytorch_cos_sim() -> None:
    """Tests the correct computation of util.pytorch_cos_scores"""
    a = np.random.randn(50, 100)
    b = np.random.randn(50, 100)

    sklearn_pairwise = sklearn.metrics.pairwise.cosine_similarity(a, b)
    pytorch_cos_scores = util.pytorch_cos_sim(a, b).numpy()
    for i in range(len(sklearn_pairwise)):
        for j in range(len(sklearn_pairwise[i])):
            assert abs(sklearn_pairwise[i][j] - pytorch_cos_scores[i][j]) < 0.001


def test_semantic_search() -> None:
    """Tests util.semantic_search function"""
    num_queries = 20
    num_k = 10

    doc_emb = torch.tensor(np.random.randn(1000, 100))
    q_emb = torch.tensor(np.random.randn(num_queries, 100))
    hits = util.semantic_search(q_emb, doc_emb, top_k=num_k, query_chunk_size=5, corpus_chunk_size=17)
    assert len(hits) == num_queries
    assert len(hits[0]) == num_k

    # Sanity Check of the results
    cos_scores = util.pytorch_cos_sim(q_emb, doc_emb)
    cos_scores_values, cos_scores_idx = cos_scores.topk(num_k)
    cos_scores_values = cos_scores_values.cpu().tolist()
    cos_scores_idx = cos_scores_idx.cpu().tolist()

    for qid in range(num_queries):
        for hit_num in range(num_k):
            assert hits[qid][hit_num]["corpus_id"] == cos_scores_idx[qid][hit_num]
            assert np.abs(hits[qid][hit_num]["score"] - cos_scores_values[qid][hit_num]) < 0.001


def test_paraphrase_mining() -> None:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = [
        "This is a test",
        "This is a test!",
        "The cat sits on mat",
        "The cat sits on the mat",
        "On the mat a cat sits",
        "A man eats pasta",
        "A woman eats pasta",
        "A man eats spaghetti",
    ]
    duplicates = util.paraphrase_mining(model, sentences)

    for score, a, b in duplicates:
        if score > 0.5:
            assert (a, b) in [(0, 1), (2, 3), (2, 4), (3, 4), (5, 6), (5, 7), (6, 7)]


def test_pairwise_cos_sim() -> None:
    a = np.random.randn(50, 100)
    b = np.random.randn(50, 100)

    # Pairwise cos
    sklearn_pairwise = 1 - sklearn.metrics.pairwise.paired_cosine_distances(a, b)
    pytorch_cos_scores = util.pairwise_cos_sim(a, b).numpy()

    assert np.allclose(sklearn_pairwise, pytorch_cos_scores)


def test_pairwise_euclidean_sim() -> None:
    a = np.array([[1, 0], [1, 1]], dtype=np.float32)
    b = np.array([[0, 0], [0, 0]], dtype=np.float32)

    euclidean_expected = np.array([-1.0, -np.sqrt(2.0)])
    euclidean_calculated = util.pairwise_euclidean_sim(a, b).numpy()

    assert np.allclose(euclidean_expected, euclidean_calculated)


def test_pairwise_manhattan_sim() -> None:
    a = np.array([[1, 0], [1, 1]], dtype=np.float32)
    b = np.array([[0, 0], [0, 0]], dtype=np.float32)

    manhattan_expected = np.array([-1.0, -2.0])
    manhattan_calculated = util.pairwise_manhattan_sim(a, b).numpy()

    assert np.allclose(manhattan_expected, manhattan_calculated)


def test_pairwise_dot_score_cos_sim() -> None:
    a = np.array([[1, 0], [1, 0], [1, 0]], dtype=np.float32)
    b = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)

    dot_and_cosine_expected = np.array([1.0, 0.0, -1.0])
    cosine_calculated = util.pairwise_cos_sim(a, b)
    dot_calculated = util.pairwise_dot_score(a, b)

    assert np.allclose(cosine_calculated, dot_and_cosine_expected)
    assert np.allclose(dot_calculated, dot_and_cosine_expected)


def test_euclidean_sim() -> None:
    a = np.array([[1, 0], [0, 1]], dtype=np.float32)
    b = np.array([[0, 0], [0, 1]], dtype=np.float32)

    euclidean_expected = np.array([[-1.0, -np.sqrt(2.0)], [-1.0, 0.0]])
    euclidean_calculated = util.euclidean_sim(a, b).detach().numpy()

    assert np.allclose(euclidean_expected, euclidean_calculated)


def test_manhattan_sim() -> None:
    a = np.array([[1, 0], [0, 1]], dtype=np.float32)
    b = np.array([[0, 0], [0, 1]], dtype=np.float32)

    manhattan_expected = np.array([[-1.0, -2.0], [-1.0, 0]])
    manhattan_calculated = util.manhattan_sim(a, b).detach().numpy()
    assert np.allclose(manhattan_expected, manhattan_calculated)


def test_dot_score_cos_sim() -> None:
    a = np.array([[1, 0]], dtype=np.float32)
    b = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)

    dot_and_cosine_expected = np.array([[1.0, 0.0, -1.0]])
    cosine_calculated = util.cos_sim(a, b)
    dot_calculated = util.dot_score(a, b)

    assert np.allclose(cosine_calculated, dot_and_cosine_expected)
    assert np.allclose(dot_calculated, dot_and_cosine_expected)
