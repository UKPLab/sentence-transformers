from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

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
    from qdrant_client import QdrantClient
    from qdrant_client.http import models

    # Validate input sparse tensors
    if not query_embeddings.is_sparse or query_embeddings.layout != torch.sparse_coo:
        raise ValueError("Query embeddings must be a sparse COO tensor")

    if corpus_index is None:
        if corpus_embeddings is None:
            raise ValueError("Either corpus_embeddings or corpus_index must be provided")

        if not corpus_embeddings.is_sparse or corpus_embeddings.layout != torch.sparse_coo:
            raise ValueError("Corpus embeddings must be a sparse COO tensor")

        # Create new Qdrant client and collection
        client = QdrantClient(url="http://localhost:6333")
        collection_name = f"sparse_collection_{int(time.time())}"

        client.create_collection(
            collection_name=f"{collection_name}",
            vectors_config={},
            sparse_vectors_config={"text": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))},
        )

        # Process and upload corpus embeddings
        corpus_indices = corpus_embeddings.coalesce().indices().cpu().numpy()
        corpus_values = corpus_embeddings.coalesce().values().cpu().numpy()

        batch_size = 10000
        num_vectors = corpus_embeddings.size(0)

        for start_idx in tqdm(range(0, num_vectors, batch_size)):
            end_idx = min(start_idx + batch_size, num_vectors)
            batch_points = []

            # Process batch
            for i in range(start_idx, end_idx):
                mask = corpus_indices[0] == i
                vector_indices = corpus_indices[1][mask]
                vector_values = corpus_values[mask]

                batch_points.append(
                    models.PointStruct(
                        id=i,
                        payload={},
                        vector={"text": models.SparseVector(indices=vector_indices, values=vector_values)},
                    )
                )

            # Upload batch
            client.upsert(collection_name=collection_name, points=batch_points)

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
            search_params=models.SearchParams(
                hnsw_ef=128,
                exact=True,  # Use exact search for better results
            ),
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
    from elasticsearch import Elasticsearch, helpers

    if not query_embeddings.is_sparse or query_embeddings.layout != torch.sparse_coo:
        raise ValueError("Query embeddings must be a sparse COO tensor")

    if corpus_index is None:
        if corpus_embeddings is None:
            raise ValueError("Either corpus_embeddings or corpus_index must be provided")

        if not corpus_embeddings.is_sparse or corpus_embeddings.layout != torch.sparse_coo:
            raise ValueError("Corpus embeddings must be a sparse COO tensor")

        es = Elasticsearch("http://localhost:9200")
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
        for start_idx in tqdm(range(0, num_docs, batch_size)):
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
