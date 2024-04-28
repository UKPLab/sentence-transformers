import os

import numpy as np
import pandas as pd
import tqdm
import json
import argparse

from james.msmarco.ms_marco_eval import load_reference, compute_metrics_james_zhao

parser = argparse.ArgumentParser()
parser.add_argument("--residing_folder", required=True, type=str)
parser.add_argument("--knn_index_file", required=True,
                    type=str, default="knn.npy")
parser.add_argument("--output_file", type=str, default="inference.tsv")
args = parser.parse_args()
print(f"{vars(args)}", flush=True)

# residing_folder = "./data/results/baseline"
# knn_index_file = "knn.npy"
# output_file = "inference.tsv"
residing_folder = args.residing_folder
knn_index_file = args.knn_index_file
output_file = args.output_file

inference_dev_ids = pd.read_csv(
    "./data/qrels/qrels_with_text.tsv", header=None, sep="\t", names=["qid", "query"])
knn_results = np.load(os.path.join(residing_folder, knn_index_file))
print(f"{len(inference_dev_ids)=}, {len(knn_results)=}", flush=True)
inference_dev_ids.head()

# qid to ranked ids
inference = {}
for qid, knn_row in zip(inference_dev_ids["qid"], knn_results):
  inference[qid] = knn_row.tolist()
reference = load_reference("./data/qrels/qrels.dev.tsv")
assert (set(inference.keys()) == set(reference.keys()))

scores, ranking = compute_metrics_james_zhao(
  qids_to_relevant_passageids=reference,
  qids_to_ranked_candidate_passages=inference,
  max_mrr_rank=1000,
)

print(scores, flush=True)
with open(os.path.join(residing_folder, "scores.json"), "w") as f:
  json.dump(scores, f)
with open(os.path.join(residing_folder, "ranking.json"), "w") as f:
  json.dump(ranking, f)
