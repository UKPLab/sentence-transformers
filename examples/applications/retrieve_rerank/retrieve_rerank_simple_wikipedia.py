"""
This examples demonstrates the setup for Query / Question-Answer-Retrieval.

You can input a query or a question. The script then uses semantic search
to find relevant passages in Simple English Wikipedia (as it is smaller and fits better in RAM).

For semantic search, we use SentenceTransformer('multi-qa-MiniLM-L6-cos-v1') and retrieve
100 potentially passages that answer the input query.

Next, we use a more powerful CrossEncoder (cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')) that
scores the query and all retrieved passages for their relevancy. The cross-encoder is neccessary to filter out certain noise
that might be retrieved from the semantic search step.

Google Colab Example: https://colab.research.google.com/github/UKPLab/sentence-transformers/blob/master/examples/applications/retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb
"""
import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import time
import gzip
import os
import torch

import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util

import gzip
import os
import torch


# We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
bi_encoder.max_seq_length = 256  # Truncate long passages to 256 tokens
top_k = 100  # Number of passages we want to retrieve with the bi-encoder

# The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# As dataset, we use Simple English Wikipedia. Compared to the full English wikipedia, it has only
# about 170k articles. We split these articles into paragraphs and encode them with the bi-encoder

wikipedia_filepath = 'simplewiki-2020-11-01.jsonl.gz'

if not os.path.exists(wikipedia_filepath):
    util.http_get('http://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz', wikipedia_filepath)

passages = []
with gzip.open(wikipedia_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        data = json.loads(line.strip())

        # Add all paragraphs
        # passages.extend(data['paragraphs'])

        # Only add the first paragraph
        passages.append(data['paragraphs'][0])

print("Passages:", len(passages))

# We encode all passages into our vector space. This takes about 5 minutes (depends on your GPU speed)
corpus_embeddings = bi_encoder.encode(passages, convert_to_tensor=True, show_progress_bar=True)


def search(query):
    print("Input question:", query)

    ##### Sematic Search #####
    # Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding.cuda()
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    # Output of top-5 hits from bi-encoder
    print("\n-------------------------\n")
    print("Top-3 Bi-Encoder Retrieval hits")
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    for hit in hits[0:3]:
        print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']].replace("\n", " ")))

    # Output of top-5 hits from re-ranker
    print("\n-------------------------\n")
    print("Top-3 Cross-Encoder Re-ranker hits")
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    for hit in hits[0:3]:
        print("\t{:.3f}\t{}".format(hit['cross-score'], passages[hit['corpus_id']].replace("\n", " ")))


while True:
    query = input("Please enter a question: ")
    search(query)
    print("\n\n========\n")