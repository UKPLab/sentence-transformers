from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from sentence_transformers import util


def create_sparse_tensor(rows, cols, num_nonzero, seed=None):
    """Create a sparse tensor of shape (rows, cols) with num_nonzero values per row."""
    if seed is not None:
        torch.manual_seed(seed)

    indices = []
    values = []

    for i in range(rows):
        row_indices = torch.stack(
            [torch.full((num_nonzero,), i, dtype=torch.long), torch.randint(0, cols, (num_nonzero,))]
        )
        row_values = torch.randn(num_nonzero)

        indices.append(row_indices)
        values.append(row_values)

    indices = torch.cat(indices, dim=1)
    values = torch.cat(values)
    return torch.sparse_coo_tensor(indices, values, (rows, cols)).coalesce()


@pytest.fixture
def sparse_tensors():
    """Create two large sparse tensors of shape (50, 100) each."""
    rows, cols = 50, 1000
    num_nonzero = 10  # per row

    tensor1 = create_sparse_tensor(rows, cols, num_nonzero, seed=42)
    tensor2 = create_sparse_tensor(rows, cols, num_nonzero, seed=1337)
    if torch.cuda.is_available():
        return tensor1.to("cuda"), tensor2.to("cuda")
    else:
        return tensor1, tensor2


def test_cos_sim_sparse(sparse_tensors):
    """Test cosine similarity between sparse and dense representations."""
    tensor1, tensor2 = sparse_tensors

    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    sim_sparse = util.cos_sim(tensor1, tensor2)
    sim_dense = util.cos_sim(dense1, dense2)

    assert torch.allclose(sim_sparse, sim_dense, rtol=1e-5, atol=1e-5)


def test_dot_score_sparse(sparse_tensors):
    """Test dot product with sparse tensors."""
    tensor1, tensor2 = sparse_tensors

    # Convert to dense before computing
    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    score_sparse = util.dot_score(tensor1, tensor2)
    score_dense = util.dot_score(dense1, dense2)

    assert torch.allclose(score_sparse, score_dense, rtol=1e-5, atol=1e-5)


def test_manhattan_sim_sparse(sparse_tensors):
    """Test Manhattan similarity with sparse tensors."""
    tensor1, tensor2 = sparse_tensors

    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    sim_sparse = util.manhattan_sim(tensor1, tensor2)
    sim_dense = util.manhattan_sim(dense1, dense2)

    assert torch.allclose(sim_sparse, sim_dense, rtol=1e-5, atol=1e-5)


def test_euclidean_sim_sparse(sparse_tensors):
    """Test Euclidean similarity with sparse tensors."""
    tensor1, tensor2 = sparse_tensors

    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    sim_sparse = util.euclidean_sim(tensor1, tensor2)
    sim_dense = util.euclidean_sim(dense1, dense2)

    assert torch.allclose(sim_sparse, sim_dense, rtol=1e-5, atol=1e-5)


def test_pairwise_cos_sim_sparse(sparse_tensors):
    """Test pairwise cosine similarity with sparse tensors."""
    tensor1, tensor2 = sparse_tensors

    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    sim_sparse = util.pairwise_cos_sim(tensor1, tensor2)
    sim_dense = util.pairwise_cos_sim(dense1, dense2)

    assert torch.allclose(sim_sparse, sim_dense, rtol=1e-5, atol=1e-5)


def test_pairwise_dot_score_sparse(sparse_tensors):
    """Test pairwise dot product with sparse tensors."""
    tensor1, tensor2 = sparse_tensors

    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    score_sparse = util.pairwise_dot_score(tensor1, tensor2)
    score_dense = util.pairwise_dot_score(dense1, dense2)

    assert torch.allclose(score_sparse, score_dense, rtol=1e-5, atol=1e-5)


def test_pairwise_manhattan_sim_sparse(sparse_tensors):
    """Test pairwise Manhattan similarity with sparse tensors."""
    tensor1, tensor2 = sparse_tensors

    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    sim_sparse = util.pairwise_manhattan_sim(tensor1, tensor2)
    sim_dense = util.pairwise_manhattan_sim(dense1, dense2)

    assert torch.allclose(sim_sparse, sim_dense, rtol=1e-5, atol=1e-5)


def test_pairwise_euclidean_sim_sparse(sparse_tensors):
    """Test pairwise Euclidean similarity with sparse tensors."""
    tensor1, tensor2 = sparse_tensors

    dense1 = tensor1.to_dense()
    dense2 = tensor2.to_dense()

    sim_sparse = util.pairwise_euclidean_sim(tensor1, tensor2)
    sim_dense = util.pairwise_euclidean_sim(dense1, dense2)

    assert torch.allclose(sim_sparse, sim_dense, rtol=1e-5, atol=1e-5)


def test_performance_with_large_vectors():
    """Test performance (time) for all similarity functions with large sparse vectors vs dense."""

    # Set dimensions for large sparse vectors
    rows = 500  # Just a few vectors to compare
    cols = 100000  # Large dimensionality
    num_nonzero = 128  # 128 non-null elements per vector

    print("\nPerformance test with large sparse vs. dense vectors")
    print(f"Shape: ({rows}, {cols}), Non-zeros per vector: {num_nonzero}")

    # Create large sparse tensors
    print("Creating sparse tensors...")
    tensor1_sparse = create_sparse_tensor(rows, cols, num_nonzero, seed=42)
    tensor2_sparse = create_sparse_tensor(rows, cols, num_nonzero, seed=1337)

    # Convert to dense for comparison
    print("Converting to dense tensors...")
    tensor1_dense = tensor1_sparse.to_dense()
    tensor2_dense = tensor2_sparse.to_dense()

    # List of functions to test
    similarity_functions = [
        ("cos_sim", util.cos_sim),
        ("dot_score", util.dot_score),
        ("manhattan_sim", util.manhattan_sim),  # Comment until the function is implemented in a fast way
        ("euclidean_sim", util.euclidean_sim),
        ("pairwise_cos_sim", util.pairwise_cos_sim),
        ("pairwise_dot_score", util.pairwise_dot_score),
        ("pairwise_manhattan_sim", util.pairwise_manhattan_sim),
        ("pairwise_euclidean_sim", util.pairwise_euclidean_sim),
    ]

    results = []

    for name, func in similarity_functions:
        # Time sparse operation
        start_time = time.time()
        _ = func(tensor1_sparse, tensor2_sparse)
        sparse_time = time.time() - start_time

        # Time dense operation
        start_time = time.time()
        _ = func(tensor1_dense, tensor2_dense)
        dense_time = time.time() - start_time

        # Calculate speedup ratio
        speedup_ratio = dense_time / sparse_time if sparse_time > 0 else float("inf")

        results.append(
            {"function": name, "sparse_time": sparse_time, "dense_time": dense_time, "speedup_ratio": speedup_ratio}
        )

    # Print results in a table
    print("\nPerformance Results:")
    print(f"{'Function':<25} | {'Sparse Time (s)':<15} | {'Dense Time (s)':<15} | {'Speedup Ratio':<15}")
    print("-" * 80)

    for r in results:
        print(
            f"{r['function']:<25} | {r['sparse_time']:<15.6f} | {r['dense_time']:<15.6f} | {r['speedup_ratio']:<15.2f}"
        )

    # Create performance summary
    sparse_time_avg = np.mean([r["sparse_time"] for r in results])
    dense_time_avg = np.mean([r["dense_time"] for r in results])
    avg_speedup = np.mean([r["speedup_ratio"] for r in results])

    print("\nAverage Performance:")
    print(f"Time - Sparse: {sparse_time_avg:.6f}s")
    print(f"Time - Dense: {dense_time_avg:.6f}s")
    print(f"Average speedup: {avg_speedup:.2f}x")

    assert sparse_time_avg < 0.1, "Sparse operations took too long!"
