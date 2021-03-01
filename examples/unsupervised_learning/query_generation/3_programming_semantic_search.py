from sentence_transformers import SentenceTransformer, util
import gzip
import json
import os

model = SentenceTransformer('output/programming-model')

docs = []

corpus_filepath = 'wiki-programmming-20210101.jsonl.gz'
if not os.path.exists(corpus_filepath):
    util.http_get('https://sbert.net/datasets/wiki-programmming-20210101.jsonl.gz', corpus_filepath)

with gzip.open(corpus_filepath, 'rt') as fIn:
    for line in fIn:
        data = json.loads(line.strip())
        title = data['title']
        for p in data['paragraphs']:
            if len(p) > 100:    #Only take paragraphs with at least 100 chars
                docs.append((title, p))

paragraph_emb = model.encode([d[1] for d in docs], convert_to_tensor=True)

print("Available Wikipedia Articles:")
print(", ".join(sorted(list(set([d[0] for d in docs])))))


while True:
    query = input("Query: ")
    query_emb = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, paragraph_emb, top_k=3)[0]

    for hit in hits:
        doc = docs[hit['corpus_id']]
        print("{:.2f}\t{}\t\t{}".format(hit['score'], doc[0], doc[1]))

    print("\n=================\n")
