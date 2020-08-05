"""
This is an interactive demonstration for information retrieval. We will encode a large corpus with 500k+ questions.
This is done once and the result is stored on disc.

Then, we can enter new questions. The new question is encoded and we perform a brute force cosine similarity search
and retrieve the top 5 questions in the corpus with the highest cosine similarity.

For larger datasets, it can make sense to use a vector index server like https://github.com/spotify/annoy or https://github.com/facebookresearch/faiss

"""

from sentence_transformers import SentenceTransformer, util
import os
from zipfile import ZipFile
import pickle
import time

model_name = 'distilbert-base-nli-stsb-quora-ranking'
embedding_cache_path = 'quora-embeddings-{}.pkl'.format(model_name.replace('/', '_'))
max_corpus_size = 100000

model = SentenceTransformer(model_name)


#Check if embedding cache path exists
if not os.path.exists(embedding_cache_path):
    # Check if the dataset exists. If not, download and extract
    dataset_path = 'quora-IR-dataset'
    if not os.path.exists(dataset_path):
        print("Dataset not found. Download")
        zip_save_path = 'quora-IR-dataset.zip'
        util.http_get(url='https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/quora-IR-dataset.zip', path=zip_save_path)
        with ZipFile(zip_save_path, 'r') as zip:
            zip.extractall(dataset_path)

    corpus_sentences = []
    with open(os.path.join(dataset_path, 'graph/sentences.tsv'), encoding='utf8') as fIn:
        next(fIn) #Skip header
        for line in fIn:
            qid, sentence = line.strip().split('\t')
            corpus_sentences.append(sentence)

            if len(corpus_sentences) >= max_corpus_size:
                break

    print("Encode the corpus. This might take a while")
    corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_tensor=True)

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

    start_time = time.time()
    question_embedding = model.encode(inp_question, convert_to_tensor=True)
    hits = util.information_retrieval(question_embedding, corpus_embeddings)
    end_time = time.time()
    hits = hits[0]  #Get the hits for the first query

    print("Input question:", inp_question)
    print("Results (after {:.3f} seconds):".format(end_time-start_time))
    for hit in hits[0:5]:
        print("\t{:.3f}\t{}".format(hit['score'], corpus_sentences[hit['corpus_id']]))

    print("\n\n========\n")