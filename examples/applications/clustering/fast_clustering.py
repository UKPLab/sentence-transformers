"""
This is a more complex example on performing clustering on large scale dataset.

This examples find in a large set of sentences local communities, i.e., groups of sentences that are highly
similar. You can freely configure the threshold what is considered as similar. A high threshold will
only find extremely similar sentences, a lower threshold will find more sentence that are less similar.

A second parameter is 'min_community_size': Only communities with at least a certain number of sentences will be returned.

The method for finding the communities is extremely fast, for clustering 50k sentences it requires only 5 seconds (plus embedding comuptation).

In this example, we download a large set of questions from Quora and then find
similar questions in this set.
"""
from sentence_transformers import SentenceTransformer, util
import os
import csv
import pickle
import time




# Model for computing sentence embeddings. We use one trained for similar questions detection
model = SentenceTransformer('distilbert-base-nli-stsb-quora-ranking')

# We donwload the Quora Duplicate Questions Dataset (https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)
# and find similar question in it
url = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
dataset_path = "quora_duplicate_questions.tsv"
max_corpus_size = 50000 # We limit our corpus to only the first 50k questions
embedding_cache_path = 'quora-embeddings-size-{}.pkl'.format(max_corpus_size)

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
    corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)

    print("Store file on disc")
    with open(embedding_cache_path, "wb") as fOut:
        pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)
else:
    print("Load pre-computed embeddings from disc")
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
        corpus_sentences = cache_data['sentences']
        corpus_embeddings = cache_data['embeddings']

print("Start clustering")
start_time = time.time()

#Two parameter to tune:
#min_cluster_size: Only consider cluster that have at least 25 elements (30 similar sentences)
#threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
clusters = util.community_detection(corpus_embeddings, min_community_size=25, threshold=0.95)


#Print all cluster / communities
for i, cluster in enumerate(clusters):
    print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
    for sentence_id in cluster:
        print("\t", corpus_sentences[sentence_id])



print("Clustering done after {:.2f} sec".format(time.time() - start_time))