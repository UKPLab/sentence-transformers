"""
This script contains an example how to perform re-ranking with a Cross-Encoder for semantic search.

First, we use an efficient Bi-Encoder to retrieve similar questions from the Quora Duplicate Questions dataset:
https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs

Then, we re-rank the hits from the Bi-Encoder using a Cross-Encoder.
"""
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
import os
import csv
import pickle
import time
import sys

# We use a BiEncoder (SentenceTransformer) that produces embeddings for questions.
# We then search for similar questions using cosine similarity and identify the top 100 most similar questions
model_name = 'distilbert-multilingual-nli-stsb-quora-ranking'
model = SentenceTransformer(model_name)
num_candidates = 500

# To refine the results, we use a CrossEncoder. A CrossEncoder gets both inputs (input_question, retrieved_question)
# and outputs a score 0...1 indicating the similarity.
cross_encoder_model = CrossEncoder('cross-encoder/roberta-base-stsb')

# Dataset we want to use
url = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
dataset_path = "quora_duplicate_questions.tsv"
max_corpus_size = 20000

# Some local file to cache computed embeddings
embedding_cache_path = 'quora-embeddings-{}-size-{}.pkl'.format(model_name.replace('/', '_'), max_corpus_size)

#Check if embedding cache path exists
if not os.path.exists(embedding_cache_path):
    # Check if the dataset exists. If not, download and extract
    # Download dataset if needed
    if not os.path.exists(dataset_path):
        print("Download dataset")
        util.http_get(url, dataset_path)

    # Get all unique sentences from the file
    corpus_sentences = set()
    with open(dataset_path, encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            corpus_sentences.add(row['question1'])
            if len(corpus_sentences) >= max_corpus_size:
                break

            corpus_sentences.add(row['question2'])
            if len(corpus_sentences) >= max_corpus_size:
                break

    corpus_sentences = list(corpus_sentences)
    print("Encode the corpus. This might take a while")
    corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_tensor=True, num_workers=2)

    print("Store file on disc")
    with open(embedding_cache_path, "wb") as fOut:
        pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)
else:
    print("Load pre-computed embeddings from disc")
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
        corpus_sentences = cache_data['sentences'][0:max_corpus_size]
        corpus_embeddings = cache_data['embeddings'][0:max_corpus_size]

###############################
print("Corpus loaded with {} sentences / embeddings".format(len(corpus_sentences)))

while True:
    inp_question = input("Please enter a question: ")
    print("Input question:", inp_question)

    #First, retrieve candidates using cosine similarity search
    start_time = time.time()
    question_embedding = model.encode(inp_question, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=num_candidates)
    hits = hits[0]  #Get the hits for the first query

    print("Cosine-Similarity search took {:.3f} seconds".format(time.time()-start_time))
    print("Top 5 hits with cosine-similarity:")
    for hit in hits[0:5]:
        print("\t{:.3f}\t{}".format(hit['score'], corpus_sentences[hit['corpus_id']]))


    #Now, do the re-ranking with the cross-encoder
    start_time = time.time()
    sentence_pairs = [[inp_question, corpus_sentences[hit['corpus_id']]] for hit in hits]
    ce_scores = cross_encoder_model.predict(sentence_pairs)

    for idx in range(len(hits)):
        hits[idx]['cross-encoder_score'] = ce_scores[idx]

    #Sort list by CrossEncoder scores
    hits = sorted(hits, key=lambda x: x['cross-encoder_score'], reverse=True)
    print("\nRe-ranking with Cross-Encoder took {:.3f} seconds".format(time.time() - start_time))
    print("Top 5 hits with CrossEncoder:")
    for hit in hits[0:5]:
        print("\t{:.3f}\t{}".format(hit['cross-encoder_score'], corpus_sentences[hit['corpus_id']]))

    print("\n\n========\n")
