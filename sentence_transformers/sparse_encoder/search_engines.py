from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch
    from qdrant_client import QdrantClient


def semantic_search_qdrant(
    query_embeddings: torch.Tensor,
    corpus_embeddings: torch.Tensor | None = None,
    corpus_index: tuple[QdrantClient, str] | None = None,
    top_k: int = 10,
    output_index: bool = False,
    **kwargs: Any,
) -> (
    tuple[list[list[dict[str, int | float]]], float]
    | tuple[list[list[dict[str, int | float]]], float, tuple[QdrantClient, str]]
):
    """
    Performs semantic search using sparse embeddings with Qdrant.

    Args:
        query_embeddings: PyTorch COO sparse tensor containing query embeddings
        corpus_embeddings: PyTorch COO sparse tensor containing corpus embeddings
            Only used if corpus_index is None
        corpus_index: Tuple of (QdrantClient, collection_name)
            If provided, uses this existing index for search
        top_k: Number of top results to retrieve
        output_index: Whether to return the Qdrant client and collection name

    Returns:
        A tuple containing:
        - List of search results in format [[{"corpus_id": int, "score": float}, ...], ...]
        - Time taken for search
        - (Optional) Tuple of (QdrantClient, collection_name) if output_index is True
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
    except ImportError:
        raise ImportError("Please install the Qdrant client with `pip install qdrant-client` to use this function.")

    # Validate input sparse tensors
    if not query_embeddings.is_sparse or query_embeddings.layout != torch.sparse_coo:
        raise ValueError("Query embeddings must be a sparse COO tensor")

    if corpus_index is None:
        if corpus_embeddings is None:
            raise ValueError("Either corpus_embeddings or corpus_index must be provided")

        if not corpus_embeddings.is_sparse or corpus_embeddings.layout != torch.sparse_coo:
            raise ValueError("Corpus embeddings must be a sparse COO tensor")

        # Create new Qdrant client and collection
        client = QdrantClient(url="http://localhost:6333", **kwargs)
        collection_name = f"sparse_collection_{int(time.time())}"

        client.create_collection(
            collection_name=f"{collection_name}",
            vectors_config={},
            sparse_vectors_config={"text": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))},
        )

        corpus = corpus_embeddings.coalesce()
        indices_arr = corpus.indices().cpu().numpy()
        values_arr = corpus.values().cpu().numpy()
        num_vectors = corpus_embeddings.size(0)
        batch_size = 10000
        vectors_batch = []
        insert_idx = 0

        # Precompute the start and end positions for each row.
        # Since the indices are sorted by row, searchsorted can be used.
        row_ids = indices_arr[0]
        starts = np.searchsorted(row_ids, np.arange(num_vectors), side="left")
        ends = np.searchsorted(row_ids, np.arange(num_vectors), side="right")

        for i in tqdm(range(num_vectors), desc="Processing and Upserting embeddings"):
            start = starts[i]
            end = ends[i]
            vec_indices = indices_arr[1][start:end].tolist()
            vec_values = values_arr[start:end].tolist()

            vector_data = {"text": models.SparseVector(indices=vec_indices, values=vec_values)}
            vectors_batch.append(vector_data)

            if len(vectors_batch) >= batch_size or i == num_vectors - 1:
                client.upload_collection(
                    collection_name=collection_name,
                    vectors=vectors_batch,
                    ids=range(insert_idx, insert_idx + len(vectors_batch)),
                )
                insert_idx += len(vectors_batch)
                vectors_batch = []

        corpus_index = (client, collection_name)

    # Extract client and collection name
    client, collection_name = corpus_index

    # Prepare results list
    all_results = []
    search_start_time = time.time()

    # Process each query
    for q_idx in range(query_embeddings.size(0)):
        # Extract query vector
        if query_embeddings.sparse_dim() == 1:
            q_indices = query_embeddings.coalesce().indices()[0].cpu().numpy().tolist()
            q_values = query_embeddings.coalesce().values().cpu().numpy().tolist()
        else:
            mask = query_embeddings.coalesce().indices()[0].cpu().numpy() == q_idx
            q_indices = query_embeddings.coalesce().indices()[1][mask].cpu().numpy().tolist()
            q_values = query_embeddings.coalesce().values()[mask].cpu().numpy().tolist()

        # Perform search
        search_results = client.query_points(
            collection_name=collection_name,
            query=models.SparseVector(indices=q_indices, values=q_values),
            limit=top_k,
            using="text",
        ).points

        # Format results
        formatted_results = [{"corpus_id": hit.id, "score": hit.score} for hit in search_results]
        all_results.append(formatted_results)

    search_time = time.time() - search_start_time

    if output_index:
        return all_results, search_time, corpus_index
    else:
        return all_results, search_time


def semantic_search_elasticsearch(
    query_embeddings: torch.Tensor,
    corpus_embeddings: torch.Tensor | None = None,
    corpus_index: tuple[Elasticsearch, str] | None = None,
    top_k: int = 10,
    output_index: bool = False,
    **kwargs: Any,
) -> (
    tuple[list[list[dict[str, int | float]]], float]
    | tuple[list[list[dict[str, int | float]]], float, tuple[Elasticsearch, str]]
):
    """
    Performs semantic search using sparse embeddings with Elasticsearch.

    Args:
        query_embeddings: PyTorch COO sparse tensor containing query embeddings
        corpus_embeddings: PyTorch COO sparse tensor containing corpus embeddings
            Only used if corpus_index is None
        corpus_index: Tuple of (Elasticsearch, collection_name)
            If provided, uses this existing index for search
        top_k: Number of top results to retrieve
        output_index: Whether to return the Elasticsearch client and collection name

    Returns:
        A tuple containing:
        - List of search results in format [[{"corpus_id": int, "score": float}, ...], ...]
        - Time taken for search
        - (Optional) Tuple of (Elasticsearch, collection_name) if output_index is True
    """
    try:
        from elasticsearch import Elasticsearch, helpers
    except ImportError:
        raise ImportError(
            "Please install the Elasticsearch client with `pip install elasticsearch` to use this function."
        )

    if not query_embeddings.is_sparse or query_embeddings.layout != torch.sparse_coo:
        raise ValueError("Query embeddings must be a sparse COO tensor")

    if corpus_index is None:
        if corpus_embeddings is None:
            raise ValueError("Either corpus_embeddings or corpus_index must be provided")

        if not corpus_embeddings.is_sparse or corpus_embeddings.layout != torch.sparse_coo:
            raise ValueError("Corpus embeddings must be a sparse COO tensor")

        es = Elasticsearch("http://localhost:9200", **kwargs)
        index_name = f"sparse_index_{int(time.time())}"

        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)

        es.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "tokens": {
                            "type": "rank_features"  # This is the key - use rank_features for sparse vectors
                        },
                        "id": {"type": "keyword"},
                    }
                }
            },
        )

        corpus = corpus_embeddings.coalesce()
        indices = corpus.indices().cpu().numpy()
        values = corpus.values().cpu().numpy()
        num_docs = corpus.size(0)

        batch_size = 1000
        for start_idx in tqdm(range(0, num_docs, batch_size), desc="Upserting embeddings"):
            end_idx = min(start_idx + batch_size, num_docs)
            actions = []

            for i in range(start_idx, end_idx):
                mask = indices[0] == i
                vec_indices = indices[1][mask]
                vec_values = values[mask]

                # Create tokens field with index->value mapping (similar to ELSER's tokens representation)
                tokens = {str(idx): float(val) for idx, val in zip(vec_indices, vec_values)}

                actions.append(
                    {
                        "_index": index_name,
                        "_id": str(i),
                        "_source": {
                            "id": str(i),
                            "tokens": tokens,  # This maps directly to rank_features
                        },
                    }
                )

            # Bulk insert the batch
            helpers.bulk(es, actions)

        es.indices.refresh(index=index_name)
        corpus_index = (es, index_name)

    es, index_name = corpus_index
    all_results = []
    search_start_time = time.time()

    for q_idx in range(query_embeddings.size(0)):
        # Extract query vector
        if query_embeddings.sparse_dim() == 1:
            q_indices = query_embeddings.coalesce().indices()[0].cpu().numpy().tolist()
            q_values = query_embeddings.coalesce().values().cpu().numpy().tolist()
        else:
            mask = query_embeddings.coalesce().indices()[0].cpu().numpy() == q_idx
            q_indices = query_embeddings.coalesce().indices()[1][mask].cpu().numpy().tolist()
            q_values = query_embeddings.coalesce().values()[mask].cpu().numpy().tolist()

        should_clauses = []
        for idx, val in zip(q_indices, q_values):
            should_clauses.append({"rank_feature": {"field": f"tokens.{idx}", "boost": float(val)}})

        # Build the actual query
        query = {"size": top_k, "query": {"bool": {"should": should_clauses, "minimum_should_match": 1}}}

        result = es.search(index=index_name, body=query)

        # Format results
        formatted = [{"corpus_id": int(hit["_id"]), "score": hit["_score"]} for hit in result["hits"]["hits"]]
        all_results.append(formatted)

    search_time = time.time() - search_start_time

    if output_index:
        return all_results, search_time, corpus_index
    else:
        return all_results, search_time
