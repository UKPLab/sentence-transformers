"""
This script contains an example how to perform semantic search with ElasticSearch.

As dataset, we use the Quora Duplicate Questions dataset, which contains about 500k questions:
https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs

Questions are indexed to ElasticSearch together with their respective sentence
embeddings.

The script shows results from BM25 as well as from semantic search with
cosine similarity.

You need ElasticSearch (https://www.elastic.co/de/elasticsearch/) up and running. Further, you need the Python
ElasticSearch Client installed: https://elasticsearch-py.readthedocs.io/en/master/

As embeddings model, we use the SBERT model 'distilbert-multilingual-nli-stsb-quora-ranking',
that it aligned for 100 languages. I.e., you can type in a question in various languages and it will
return the closest questions in the corpus (questions in the corpus are mainly in English).
"""

from sentence_transformers import SentenceTransformer, util
import os
from elasticsearch import Elasticsearch
import csv
import time
import tqdm.autonotebook
from elasticsearch import helpers

if __name__ == '__main__':
    es = Elasticsearch()

    model = SentenceTransformer('distilbert-multilingual-nli-stsb-quora-ranking')

    url = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
    dataset_path = "quora_duplicate_questions.tsv"
    max_corpus_size = 100000

    #Download dataset if needed
    if not os.path.exists(dataset_path):
        print("Download dataset")
        util.http_get(url, dataset_path)

    #Get all unique sentences from the file
    all_questions = {}
    with open(dataset_path, encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            all_questions[row['qid1']] = row['question1']
            if len(all_questions) >= max_corpus_size:
                break

            all_questions[row['qid2']] = row['question2']
            if len(all_questions) >= max_corpus_size:
                break

    qids = list(all_questions.keys())
    questions = [all_questions[qid] for qid in qids]

    #Index data, if the index does not exists
    if not es.indices.exists(index="quora"):
        try:
            es_index = {
                "mappings": {
                  "properties": {
                    "question": {
                      "type": "text"
                    },
                    "question_vector": {
                      "type": "dense_vector",
                      "dims": 768
                    }
                  }
                }
            }

            es.indices.create(index='quora', body=es_index, ignore=[400])
            chunk_size = 500
            print("Index data (you can stop it by pressing Ctrl+C once):")
            with tqdm.tqdm(total=len(qids)) as pbar:
                for start_idx in range(0, len(qids), chunk_size):
                    end_idx = start_idx+chunk_size

                    embeddings = model.encode(questions[start_idx:end_idx], show_progress_bar=False)
                    bulk_data = []
                    for qid, question, embedding in zip(qids[start_idx:end_idx], questions[start_idx:end_idx], embeddings):
                        bulk_data.append({
                                "_index": 'quora',
                                "_id": qid,
                                "_source": {
                                    "question": question,
                                    "question_vector": embedding
                                }
                            })

                    helpers.bulk(es, bulk_data)
                    pbar.update(chunk_size)

        except:
            print("During index an exception occured. Continue\n\n")




    #Interactive search queries
    while True:
        inp_question = input("Please enter a question: ")

        encode_start_time = time.time()
        question_embedding = model.encode(inp_question)
        encode_end_time = time.time()

        #Lexical search
        bm25 = es.search(index="quora", body={"query": {"match": {"question": inp_question }}})

        #Sematic search
        sem_search = es.search(index="quora", body={
              "query": {
                "script_score": {
                  "query": {
                    "match_all": {}
                  },
                  "script": {
                    "source": "cosineSimilarity(params.queryVector, doc['question_vector']) + 1.0",
                    "params": {
                      "queryVector": question_embedding
                    }
                  }
                }
              }
            })

        print("Input question:", inp_question)
        print("Computing the embedding took {:.3f} seconds, BM25 search took {:.3f} seconds, semantic search with ES took {:.3f} seconds".format(encode_end_time-encode_start_time, bm25['took']/1000, sem_search['took']/1000))

        print("BM25 results:")
        for hit in bm25['hits']['hits'][0:5]:
            print("\t{}".format(hit['_source']['question']))

        print("\nSemantic Search results:")
        for hit in sem_search['hits']['hits'][0:5]:
            print("\t{}".format(hit['_source']['question']))

        print("\n\n========\n")