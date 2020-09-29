import requests
from torch import Tensor, device
from typing import List
from tqdm import tqdm
import sys
import importlib
import os
import torch
import numpy as np
import queue


def pytorch_cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    This function can be used as a faster replacement for 1-scipy.spatial.distance.cdist(a,b)
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1))



def paraphrase_mining(model,
                      sentences: List[str],
                      show_progress_bar=False,
                      batch_size=32,
                      query_chunk_size: int = 5000,
                      corpus_chunk_size: int = 100000,
                      max_pairs: int = 500000,
                      top_k: int = 100):
    """
    Given a list of sentences / texts, this function performs paraphrase mining. It compares all sentences against all
    other sentences and returns a list with the pairs that have the highest cosine similarity score.

    :param model: SentenceTransformer model for embedding computation
    :param sentences: A list of strings (texts or sentences)
    :param show_progress_bar: Plotting of a progress bar
    :param batch_size: Number of texts that are encoded simultaneously by the model
    :param query_chunk_size: Search for most similar pairs for #query_chunk_size at the same time. Decrease, to lower memory footprint (increases run-time).
    :param corpus_chunk_size: Compare a sentence simultaneously against #corpus_chunk_size other sentences. Decrease, to lower memory footprint (increases run-time).
    :param max_pairs: Maximal number of text pairs returned.
    :param top_k: For each sentence, we retrieve up to top_k other sentences
    :return: Returns a list of triplets with the format [score, id1, id2]
    """

    top_k += 1  #A sentence has the highest similarity to itself. Increase +1 as we are interest in distinct pairs

    # Compute embedding for the sentences
    embeddings = model.encode(sentences, show_progress_bar=show_progress_bar, batch_size=batch_size,
                              convert_to_tensor=True)

    # Mine for duplicates
    pairs = queue.PriorityQueue()
    min_score = -1
    num_added = 0

    for corpus_start_idx in range(0, len(embeddings), corpus_chunk_size):
        corpus_end_idx = min(corpus_start_idx + corpus_chunk_size, len(embeddings))
        for query_start_idx in range(0, len(embeddings), query_chunk_size):
            query_end_idx = min(query_start_idx + query_chunk_size, len(embeddings))

            #logging.info("Compute cosine similarities")
            cos_scores = pytorch_cos_sim(embeddings[query_start_idx:query_end_idx],
                                         embeddings[corpus_start_idx:corpus_end_idx]).cpu()


            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k, len(cos_scores[0])-1), dim=1, largest=True, sorted=False)
            cos_scores_top_k_values = cos_scores_top_k_values.tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.tolist()

            #logging.info("Find most similar pairs out of {} queries".format(len(cos_scores)))
            for query_itr in range(len(cos_scores)):
                for top_k_idx, corpus_itr in enumerate(cos_scores_top_k_idx[query_itr]):
                    i = query_start_idx + query_itr
                    j = corpus_start_idx + corpus_itr

                    if i != j and cos_scores_top_k_values[query_itr][top_k_idx] > min_score:
                        pairs.put((cos_scores_top_k_values[query_itr][top_k_idx], i, j))
                        num_added += 1

                        if num_added >= max_pairs:
                            entry = pairs.get()
                            min_score = entry[0]

    # Get the pairs
    added_pairs = set()  # Used for duplicate detection
    pairs_list = []
    while not pairs.empty():
        score, i, j = pairs.get()
        sorted_i, sorted_j = sorted([i, j])

        if sorted_i != sorted_j and (sorted_i, sorted_j) not in added_pairs:
            added_pairs.add((sorted_i, sorted_j))
            pairs_list.append([score, i, j])

    # Highest scores first
    pairs_list = sorted(pairs_list, key=lambda x: x[0], reverse=True)
    return pairs_list


def information_retrieval(*args, **kwargs):
    """This function is decprecated. Use semantic_search insted"""
    return semantic_search(*args, **kwargs)


def semantic_search(query_embeddings: Tensor,
                      corpus_embeddings: Tensor,
                      query_chunk_size: int = 100,
                      corpus_chunk_size: int = 100000,
                      top_k: int = 10):
    """
    This function performs a cosine similarity search between a list of query embeddings  and a list of corpus embeddings.
    It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.

    :param query_embeddings: A 2 dimensional tensor with the query embeddings.
    :param corpus_embeddings: A 2 dimensional tensor with the corpus embeddings.
    :param query_chunk_size: Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory.
    :param corpus_chunk_size: Scans the corpus 100k entries at a time. Increasing that value increases the speed, but requires more memory.
    :param top_k: Retrieve top k matching entries. Note, if your corpus is larger than query_chunk_size, |Chunks|*top_k are returned
    :return: Returns a sorted list with decreasing cosine similarity scores. Entries are dictionaries with the keys 'corpus_id' and 'score'
    """

    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)

    #Normalize scores, so that the dot-product is equivalent to cosine similarity
    query_embeddings = query_embeddings / query_embeddings.norm(dim=1)[:, None]
    corpus_embeddings = corpus_embeddings / corpus_embeddings.norm(dim=1)[:, None]

    queries_result_list = [[] for _ in range(len(query_embeddings))]

    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
        query_end_idx = min(query_start_idx + query_chunk_size, len(query_embeddings))

        # Iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            corpus_end_idx = min(corpus_start_idx + corpus_chunk_size, len(corpus_embeddings))

            # Compute cosine similarites
            cos_scores = torch.mm(query_embeddings[query_start_idx:query_end_idx], corpus_embeddings[corpus_start_idx:corpus_end_idx].transpose(0, 1)).cpu().numpy()
            cos_scores = np.nan_to_num(cos_scores)

            # Partial sort scores
            cos_score_argpartition = np.argpartition(-cos_scores, min(top_k, len(cos_scores[0])-1))[:, 0:top_k]

            for query_itr in range(len(cos_scores)):
                for sub_corpus_id in cos_score_argpartition[query_itr]:
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    score = cos_scores[query_itr][sub_corpus_id]
                    queries_result_list[query_id].append({'corpus_id': corpus_id, 'score': score})

    #Sort and strip to top_k results
    for idx in range(len(queries_result_list)):
        queries_result_list[idx] = sorted(queries_result_list[idx], key=lambda x: x['score'], reverse=True)
        queries_result_list[idx] = queries_result_list[idx][0:top_k]

    return queries_result_list


def http_get(url, path):
    """
    Downloads a URL to a given path on disc
    """
    if os.path.dirname(path) != '':
        os.makedirs(os.path.dirname(path), exist_ok=True)

    req = requests.get(url, stream=True)
    if req.status_code != 200:
        print("Exception when trying to download {}. Response {}".format(url, req.status_code), file=sys.stderr)
        req.raise_for_status()
        return

    download_filepath = path+"_part"
    with open(download_filepath, "wb") as file_binary:
        content_length = req.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                progress.update(len(chunk))
                file_binary.write(chunk)

    os.rename(download_filepath, path)
    progress.close()


def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    features = batch['features']
    for paired_sentence_idx in range(len(features)):
        for feature_name in features[paired_sentence_idx]:
            features[paired_sentence_idx][feature_name] = features[paired_sentence_idx][feature_name].to(target_device)

    labels = batch['labels'].to(target_device)
    return features, labels


def fullname(o):
  """
  Gives a full name (package_name.class_name) for a class / object in Python. Will
  be used to load the correct classes from JSON files
  """

  module = o.__class__.__module__
  if module is None or module == str.__class__.__module__:
    return o.__class__.__name__  # Avoid reporting __builtin__
  else:
    return module + '.' + o.__class__.__name__

def import_from_string(dotted_path):
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    """
    try:
        module_path, class_name = dotted_path.rsplit('.', 1)
    except ValueError:
        msg = "%s doesn't look like a module path" % dotted_path
        raise ImportError(msg)

    module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = 'Module "%s" does not define a "%s" attribute/class' % (module_path, class_name)
        raise ImportError(msg)