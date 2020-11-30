"""
This examples demonstrates the setup for Query / Question-Answer-Retrieval.

You can input a query or a question. The script then uses semantic search
to find relevant passages in Simple English Wikipedia (as it is smaller and fits better in RAM).

For semantic search, we use SentenceTransformer('distilroberta-base-msmarco-v2') and retrieve
100 potentially passages that answer the input query.

Next, we use a more powerful CrossEncoder (cross_encoder = CrossEncoder('sentence-transformers/ce-ms-marco-TinyBERT-L-6')) that
scores the query and all retrieved passages for their relevancy. The cross-encoder is neccessary to filter out certain noise
that might be retrieved from the semantic search step.
"""
import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import time
import gzip
import os

#We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
bi_encoder = SentenceTransformer('distilroberta-base-msmarco-v2')
top_k = 100     #Number of passages we want to retrieve with the bi-encoder

#The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
cross_encoder = CrossEncoder('sentence-transformers/ce-ms-marco-TinyBERT-L-6')

# As dataset, we use Simple English Wikipedia. Compared to the full English wikipedia, it has only
# about 170k articles. We split these articles into paragraphs and encode them with the bi-encoder

wikipedia_filepath = 'data/simplewiki-2020-11-01.jsonl.gz'

if not os.path.exists(wikipedia_filepath):
    util.http_get('http://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz', wikipedia_filepath)

passages = []
with gzip.open(wikipedia_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        data = json.loads(line.strip())
        paragraphs = data['text'].split("\n\n")
        for p in paragraphs:
            if len(p.strip()) > 20:
                passages.append(p.strip()[0:5000])

#If you like, you can also limit the number of passages you want to use
#passages = passages[0:50000]
print("Passages:", len(passages))

#Now we encode all passages we have in our Simple Wikipedia corpus
corpus_embeddings = bi_encoder.encode(passages, show_progress_bar=True)

while True:
    query = input("Please enter a question: ")

    #Encode the query using the bi-encoder and find potentially relevant passages
    start_time = time.time()
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query
    print("Hits", len(hits))

    #Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    #Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    end_time = time.time()

    #Output of top-5 hits
    print("Input question:", query)
    print("Results (after {:.3f} seconds):".format(end_time - start_time))
    for hit in hits[0:5]:
        print("\t{:.3f}\t{}".format(hit['cross-score'], passages[hit['corpus_id']]))

    print("\n\n========\n")