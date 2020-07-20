from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import batch_to_device
import torch
import numpy as np
import logging
import os
import csv


class MSEEvaluatorFromDataLoader(SentenceEvaluator):
    """
    Computes the mean squared error (x100) between the computed sentence embedding
    and some target sentence embedding
    """
    def __init__(self, dataloader, name=''):
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name

        if name:
            name = "_"+name
        self.csv_file = "mse_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "MSE"]

    def __call__(self, model, output_path, epoch  = -1, steps = -1):
        model.eval()
        self.dataloader.collate_fn = model.smart_batching_collate

        embeddings = []
        labels = []
        for step, batch in enumerate(self.dataloader):
            features, batch_labels = batch_to_device(batch, self.device)
            with torch.no_grad():
                emb1 = model(features[0])['sentence_embedding'].to("cpu").numpy()

            labels.extend(batch_labels.to("cpu").numpy())
            embeddings.extend(emb1)

        embeddings = np.asarray(embeddings)
        labels = np.asarray(labels)

        mse = ((embeddings - labels)**2).mean()

        logging.info("MSE evaluation on "+self.name+" dataset")
        mse *= 100

        logging.info("embeddings shape:\t"+str(embeddings.shape))
        logging.info("MSE (*100):\t{:4f}".format(mse))

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, mse])


        return -mse #Return negative score as SentenceTransformers maximizes the performance