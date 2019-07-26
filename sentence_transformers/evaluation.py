from enum import Enum
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import csv
import logging
import os
import numpy as np
from typing import Iterable

from .models import TransformerModel
from .util import batch_to_device


class SentenceEvaluator:
    """
    Base class for all evaluators

    Extend this class and implement __call__ for custom evaluators.
    """

    def __call__(self, model: TransformerModel, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        """
        This is called during training to evaluate the model.
        It returns a score for the evaluation with a higher score indicating a better result.

        :param model:
            the model to evaluate
        :param output_path:
            path where predictions and metrics are written to
        :param epoch
            the epoch where the evaluation takes place.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation on test data.
        :param steps
            the steps in the current epoch at time of the evaluation.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation at the end of the epoch.
        :return: a score for the evaluation with a higher score indicating a better result
        """
        pass


class SequentialEvaluator(SentenceEvaluator):
    """
    This evaluator allows that multiple sub-evaluators are passed. When the model is evaluated,
    the data is passed sequentially to all sub-evaluators.

    The score from the last sub-evaluator will be used as the main score for the best model decision.
    """
    def __init__(self, evaluators: Iterable[SentenceEvaluator]):
        self.evaluators = evaluators

    def __call__(self, model: TransformerModel, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        for evaluator in self.evaluators:
            main_score = evaluator(model, output_path, epoch, steps)

        return main_score


class LabelAccuracyEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on its accuracy on a labeled dataset

    This requires a model with LossFunction.SOFTMAX

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """
    dataloader: DataLoader

    def __init__(self, dataloader: DataLoader, name: str = ""):
        """
        Constructs an evaluator for the given dataset

        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        if name:
            name = "_"+name

        self.csv_file = "accuracy_evaluation"+name+"_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy"]

    def __call__(self, model: TransformerModel, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        total = 0
        correct = 0

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logging.info("Evaluation on the "+self.name+" dataset"+out_txt)
        for step, batch in enumerate(tqdm(self.dataloader, desc="Evaluating")):
            batch = batch_to_device(batch, self.device)
            input_ids, segment_ids, input_masks, label_ids = batch
            with torch.no_grad():
                _, prediction = model(input_ids, segment_ids, input_masks)

            total += prediction.size(0)
            correct += torch.argmax(prediction, dim=1).eq(label_ids).sum().item()
        accuracy = correct/total

        logging.info("Accuracy: {:.4f} ({}/{})\n".format(accuracy, correct, total))

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy])
            else:
                with open(csv_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy])

        return accuracy


class EmbeddingSimilarity(Enum):
    COSINE = 0
    EUCLIDEAN = 1
    MANHATTAN = 2


class EmbeddingSimilarityEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the Spearman correlation with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """
    dataloader: DataLoader
    main_similarity: EmbeddingSimilarity


    def __init__(self, dataloader: DataLoader, main_similarity: EmbeddingSimilarity = None, name:str =''):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param dataloader:
            the data for the evaluation
        :param main_similarity:
            the similarity metric that will be used for the returned score
        """
        self.dataloader = dataloader
        self.main_similarity = main_similarity
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        if name:
            name = "_"+name

        self.csv_file: str = "similarity_evaluation"+name+"_results.csv"
        self.csv_headers = ["epoch", "steps", "cosine_pearson", "cosine_spearman", "euclidean_pearson", "euclidean_spearman", "manhattan_pearson", "manhattan_spearman"]

    def __call__(self, model: TransformerModel, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        embeddings1 = []
        embeddings2 = []
        labels = []

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logging.info("Evaluation the model on "+self.name+" dataset"+out_txt)
        for step, batch in enumerate(tqdm(self.dataloader, desc="Evaluating")):
            batch = batch_to_device(batch, self.device)
            input_ids, segment_ids, input_masks, label_ids = batch
            with torch.no_grad():
                emb1 = model.get_sentence_representation(input_ids[0], segment_ids[0], input_masks[0]).to("cpu").numpy()
                emb2 = model.get_sentence_representation(input_ids[1], segment_ids[1], input_masks[1]).to("cpu").numpy()
            labels.extend(label_ids.to("cpu").numpy())
            embeddings1.extend(emb1)
            embeddings2.extend(emb2)
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)

        eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
        eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

        eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
        eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

        eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
        eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

        logging.info("Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:4f}".format(
            eval_pearson_cosine, eval_spearman_cosine))
        logging.info("Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:4f}".format(
            eval_pearson_manhattan, eval_spearman_manhattan))
        logging.info("Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:4f}".format(
            eval_pearson_euclidean, eval_spearman_euclidean))

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, eval_pearson_cosine, eval_spearman_cosine, eval_pearson_euclidean,
                                     eval_spearman_euclidean, eval_pearson_manhattan, eval_spearman_manhattan])
            else:
                with open(csv_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, eval_pearson_cosine, eval_spearman_cosine, eval_pearson_euclidean,
                                     eval_spearman_euclidean, eval_pearson_manhattan, eval_spearman_manhattan])

        if self.main_similarity == EmbeddingSimilarity.COSINE:
            return eval_spearman_cosine
        elif self.main_similarity == EmbeddingSimilarity.EUCLIDEAN:
            return eval_spearman_euclidean
        elif self.main_similarity == EmbeddingSimilarity.MANHATTAN:
            return eval_spearman_manhattan
        elif self.main_similarity is None:
            return max(eval_spearman_cosine, eval_spearman_manhattan, eval_spearman_euclidean)
        else:
            raise ValueError("Unknown main_similarity value")


class BinaryEmbeddingSimilarityEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the accuracy of identifying similar and
    dissimilar sentences.
    This is done by taking the metrics and checking if sentence pairs with a label of 0 are in the top 50% and pairs
    with label 1 in the bottom 50%.
    This assumes that the dataset is split 50-50.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the accuracy with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """
    dataloader: DataLoader
    main_similarity: EmbeddingSimilarity


    def __init__(self, dataloader: DataLoader,
                 main_similarity: EmbeddingSimilarity = EmbeddingSimilarity.COSINE, name:str =''):
        """
        Constructs an evaluator based for the dataset

        The labels need to be 0 for dissimilar pairs and 1 for similar pairs.
        The dataset needs to be split 50-50 with the labels.

        :param dataloader:
            the data for the evaluation
        :param main_similarity:
            the similarity metric that will be used for the returned score
        """
        self.dataloader = dataloader
        self.main_similarity = main_similarity
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        if name:
            name = "_"+name

        self.csv_file: str = "binary_similarity_evaluation"+name+"_results.csv"
        self.csv_headers = ["epoch", "steps", "cosine_acc", "euclidean_acc", "manhattan_acc"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        embeddings1 = []
        embeddings2 = []
        labels = []

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logging.info("Evaluation the model on "+self.name+" dataset"+out_txt)
        for step, batch in enumerate(tqdm(self.dataloader, desc="Evaluating")):
            batch = batch_to_device(batch, self.device)
            input_ids, segment_ids, input_masks, label_ids = batch
            with torch.no_grad():
                emb1 = model.get_sentence_representation(input_ids[0], segment_ids[0], input_masks[0]).to("cpu").numpy()
                emb2 = model.get_sentence_representation(input_ids[1], segment_ids[1], input_masks[1]).to("cpu").numpy()
            labels.extend(label_ids.to("cpu").numpy())
            embeddings1.extend(emb1)
            embeddings2.extend(emb2)
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)

        #Ensure labels are just 0 or 1
        for label in labels:
            assert (label == 0 or label == 1)

        cosine_middle = np.median(cosine_scores)
        cosine_acc = 0
        for label, score in zip(labels, cosine_scores):
            if (label == 1 and score > cosine_middle) or (label == 0 and score <= cosine_middle):
                cosine_acc += 1
        cosine_acc /= len(labels)

        manhattan_middle = np.median(manhattan_distances)
        manhattan_acc = 0
        for label, score in zip(labels, manhattan_distances):
            if (label == 1 and score > manhattan_middle) or (label == 0 and score <= manhattan_middle):
                manhattan_acc += 1
        manhattan_acc /= len(labels)

        euclidean_middle = np.median(euclidean_distances)
        euclidean_acc = 0
        for label, score in zip(labels, euclidean_distances):
            if (label == 1 and score > euclidean_middle) or (label == 0 and score <= euclidean_middle):
                euclidean_acc += 1
        euclidean_acc /= len(labels)

        logging.info("Cosine-Classification:\t{:4f}".format(
            cosine_acc))
        logging.info("Manhattan-Classification:\t{:4f}".format(
            manhattan_acc))
        logging.info("Euclidean-Classification:\t{:4f}\n".format(
            euclidean_acc))

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, cosine_acc, euclidean_acc, manhattan_acc])
            else:
                with open(csv_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, cosine_acc, euclidean_acc, manhattan_acc])

        if self.main_similarity == EmbeddingSimilarity.COSINE:
            return cosine_acc
        elif self.main_similarity == EmbeddingSimilarity.EUCLIDEAN:
            return euclidean_acc
        elif self.main_similarity == EmbeddingSimilarity.MANHATTAN:
            return manhattan_acc
        else:
            raise ValueError("Unknown main_similarity value")


class TripletEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on a triplet: (sentence, positive_example, negative_example). Checks if distance(sentence,positive_example) < distance(sentence, negative_example).

    """
    dataloader: DataLoader
    main_similarity: EmbeddingSimilarity


    def __init__(self, dataloader: DataLoader, main_distance_function: EmbeddingSimilarity = None, name: str =''):
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

    def __call__(self, model: TransformerModel, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
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

        for step, batch in enumerate(tqdm(self.dataloader, desc="Evaluating")):
            batch = batch_to_device(batch, self.device)
            input_ids, segment_ids, input_masks, label_ids = batch
            with torch.no_grad():
                emb1 = model.get_sentence_representation(input_ids[0], segment_ids[0], input_masks[0]).to("cpu").numpy()
                emb2 = model.get_sentence_representation(input_ids[1], segment_ids[1], input_masks[1]).to("cpu").numpy()
                emb3 = model.get_sentence_representation(input_ids[2], segment_ids[2], input_masks[2]).to("cpu").numpy()



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

        if self.main_distance_function == EmbeddingSimilarity.COSINE:
            return accuracy_cos
        if self.main_distance_function == EmbeddingSimilarity.MANHATTAN:
            return accuracy_manhatten
        if self.main_distance_function == EmbeddingSimilarity.EUCLIDEAN:
            return accuracy_euclidean

        return max(accuracy_cos, accuracy_manhatten, accuracy_euclidean)