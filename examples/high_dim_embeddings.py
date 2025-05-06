from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
texts = ["Wikipedia paragraph about particle physics"] * 1000

# Batch processing with chunking
batch_size = 32
embeddings = []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    embeddings.extend(model.encode(batch))
    
print(f"Generated {len(embeddings)} vectors of dim {len(embeddings[0])}")
