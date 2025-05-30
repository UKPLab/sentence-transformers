from sentence_transformers.models import Router
from sentence_transformers.sparse_encoder import SparseEncoder
from sentence_transformers.sparse_encoder.models import IDF, MLMTransformer, SpladePooling

print("# ------------------------------------------example with v2 distill-----------------------------------------")
doc_encoder = MLMTransformer("opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill")
router = Router.for_query_document(
    query_modules=[
        IDF.from_json(
            "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
            tokenizer=doc_encoder.tokenizer,
            frozen=True,
        ),
    ],
    document_modules=[
        doc_encoder,
        SpladePooling("max"),
    ],
)

model = SparseEncoder(
    modules=[router],
    similarity_fn_name="dot",
)

query = "What's the weather in ny now?"
document = "Currently New York is rainy."

query_embed = model.encode_query(query)
document_embed = model.encode_document(document)

sim = model.similarity(query_embed, document_embed)
print(f"Similarity: {sim}")

# Visualize top tokens for each text
top_k = 3
print(f"\nTop tokens {top_k} for each text:")

decoded_query = model.decode(query_embed, top_k=top_k)
decoded_document = model.decode(document_embed)

for i in range(top_k):
    query_token, query_score = decoded_query[i]
    doc_score = next((score for token, score in decoded_document if token == query_token), 0)
    if doc_score != 0:
        print(f"Token: {query_token}, Query score: {query_score:.4f}, Document score: {doc_score:.4f}")

"""
# ------------------------------------------example with v2 distill-----------------------------------------
Similarity: tensor([[17.5307]], device='cuda:0')

Top tokens 3 for each text:
Token: ny, Query score: 5.7729, Document score: 1.4109
Token: weather, Query score: 4.5684, Document score: 1.4673
Token: now, Query score: 3.5895, Document score: 0.7473
"""

print("# -----------------------------------------example with v3 distill-----------------------------------------")
doc_encoder = MLMTransformer("opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill")
router = Router.for_query_document(
    query_modules=[
        IDF.from_json(
            "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill",
            tokenizer=doc_encoder.tokenizer,
            frozen=True,
        ),
    ],
    document_modules=[
        doc_encoder,
        SpladePooling(pooling_strategy="max", activation_function="log1p_relu"),
    ],
)

model = SparseEncoder(
    modules=[router],
    similarity_fn_name="dot",
)

query = "What's the weather in ny now?"
document = "Currently New York is rainy."

query_embed = model.encode_query(query)
document_embed = model.encode_document(document)

sim = model.similarity(query_embed, document_embed)
print(f"Similarity: {sim}")

# Visualize top tokens for each text
top_k = 10
print(f"\nTop tokens {top_k} for each text:")

decoded_query = model.decode(query_embed, top_k=top_k)
decoded_document = model.decode(document_embed)

for i in range(min(top_k, len(decoded_query))):
    query_token, query_score = decoded_query[i]
    doc_score = next((score for token, score in decoded_document if token == query_token), 0)
    if doc_score != 0:
        print(f"Token: {query_token}, Query score: {query_score:.4f}, Document score: {doc_score:.4f}")

"""
# -----------------------------------------example with v3 distill-----------------------------------------
Similarity: tensor([[11.1105]], device='cuda:0')

Top tokens 10 for each text:
Token: ny, Query score: 5.7729, Document score: 0.8049
Token: weather, Query score: 4.5684, Document score: 0.9710
Token: now, Query score: 3.5895, Document score: 0.4720
Token: ?, Query score: 3.3313, Document score: 0.0286
Token: what, Query score: 2.7699, Document score: 0.0787
Token: in, Query score: 0.4989, Document score: 0.0417
"""
