from sentence_transformers import SentenceTransformer

def test_high_dim_embeddings():
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    emb = model.encode("Test text")
    assert len(emb) == 1024, "High-dim model output validation failed"
