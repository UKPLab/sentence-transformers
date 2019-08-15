from . import SentenceEvaluator, SimilarityFunction
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from ..util import batch_to_device
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances



class TripletEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on a triplet: (sentence, positive_example, negative_example). Checks if distance(sentence,positive_example) < distance(sentence, negative_example).
    """
    def __init__(self, dataloader: DataLoader, main_distance_function: SimilarityFunction = None, name: str =''):
        """
        Constructs an evaluator based for the dataset


        :param dataloader:
            the data for the evaluation
        :param main_similarity:
            the similarity metric that will be used for the returned score
        """
        self.dataloader = dataloader
        self.main_distance_function = main_distance_function
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        if name:
            name = "_"+name

        self.csv_file: str = "triplet_evaluation"+name+"_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy_cosinus", "accuracy_manhatten", "accuracy_euclidean"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logging.info("Evaluation the model on "+self.name+" dataset"+out_txt)

        num_triplets = 0
        num_correct_cos_triplets, num_correct_manhatten_triplets, num_correct_euclidean_triplets = 0, 0, 0

        self.dataloader.collate_fn = model.smart_batching_collate
        for step, batch in enumerate(tqdm(self.dataloader, desc="Evaluating")):
            features, label_ids = batch_to_device(batch, self.device)
            with torch.no_grad():
                emb1, emb2, emb3 = [model(sent_features)['sentence_embedding'].to("cpu").numpy() for sent_features in features]

            #Cosine distance
            pos_cos_distance = paired_cosine_distances(emb1, emb2)
            neg_cos_distances = paired_cosine_distances(emb1, emb3)

            # Manhatten
            pos_manhatten_distance = paired_manhattan_distances(emb1, emb2)
            neg_manhatten_distances = paired_manhattan_distances(emb1, emb3)

            # Euclidean
            pos_euclidean_distance = paired_euclidean_distances(emb1, emb2)
            neg_euclidean_distances = paired_euclidean_distances(emb1, emb3)

            for idx in range(len(pos_cos_distance)):
                num_triplets += 1

                if pos_cos_distance[idx] < neg_cos_distances[idx]:
                    num_correct_cos_triplets += 1

                if pos_manhatten_distance[idx] < neg_manhatten_distances[idx]:
                    num_correct_manhatten_triplets += 1

                if pos_euclidean_distance[idx] < neg_euclidean_distances[idx]:
                    num_correct_euclidean_triplets += 1



        accuracy_cos = num_correct_cos_triplets / num_triplets
        accuracy_manhatten = num_correct_manhatten_triplets / num_triplets
        accuracy_euclidean = num_correct_euclidean_triplets / num_triplets

        logging.info("Accuracy Cosine Distance:\t{:.4f}".format(accuracy_cos))
        logging.info("Accuracy Manhatten Distance:\t{:.4f}".format(accuracy_manhatten))
        logging.info("Accuracy Euclidean Distance:\t{:.4f}\n".format(accuracy_euclidean))

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy_cos, accuracy_manhatten, accuracy_euclidean])

            else:
                with open(csv_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy_cos, accuracy_manhatten, accuracy_euclidean])

        if self.main_distance_function == SimilarityFunction.COSINE:
            return accuracy_cos
        if self.main_distance_function == SimilarityFunction.MANHATTAN:
            return accuracy_manhatten
        if self.main_distance_function == SimilarityFunction.EUCLIDEAN:
            return accuracy_euclidean

        return max(accuracy_cos, accuracy_manhatten, accuracy_euclidean)