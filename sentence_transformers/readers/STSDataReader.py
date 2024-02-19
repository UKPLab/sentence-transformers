from . import InputExample
import csv
import gzip
import os


class STSDataReader:
    """
    Reads in the STS dataset. Each line contains two sentences (s1_col_idx, s2_col_idx) and one label (score_col_idx)

    Default values expects a tab separated file with the first & second column the sentence pair and third column the score (0...1). Default config normalizes scores from 0...5 to 0...1
    """

    def __init__(
        self,
        dataset_folder,
        s1_col_idx=0,
        s2_col_idx=1,
        score_col_idx=2,
        delimiter="\t",
        quoting=csv.QUOTE_NONE,
        normalize_scores=True,
        min_score=0,
        max_score=5,
    ):
        """
        Initializes the STSDataReader.

        Parameters
        ----------
        dataset_folder : str
            The folder containing the STS dataset files.

        s1_col_idx : int, optional
            Index of the column containing the first sentence. Default is 0.

        s2_col_idx : int, optional
            Index of the column containing the second sentence. Default is 1.

        score_col_idx : int, optional
            Index of the column containing the score. Default is 2.

        delimiter : str, optional
            The delimiter used in the dataset files. Default is "\t".

        quoting : int, optional
            The quoting style used in the csv files. Default is csv.QUOTE_NONE.

        normalize_scores : bool, optional
            Whether to normalize scores to a range of 0...1. Default is True.

        min_score : int, optional
            The minimum score in the dataset. Default is 0.

        max_score : int, optional
            The maximum score in the dataset. Default is 5.
        """
        self.dataset_folder = dataset_folder
        self.score_col_idx = score_col_idx
        self.s1_col_idx = s1_col_idx
        self.s2_col_idx = s2_col_idx
        self.delimiter = delimiter
        self.quoting = quoting
        self.normalize_scores = normalize_scores
        self.min_score = min_score
        self.max_score = max_score

    def get_examples(self, filename:str, max_examples:int=0) -> list:
        """
        Reads examples from the STS dataset.

        Parameters
        ----------
        filename : str
            The name of the data split file to read (train.csv, dev.csv, test.csv).

        max_examples : int, optional
            Maximum number of examples to read. Default is 0, meaning read all examples.

        Returns
        -------
        examples : list
            A list of InputExample objects.
        """
        filepath = os.path.join(self.dataset_folder, filename)
        with gzip.open(filepath, "rt", encoding="utf8") if filename.endswith(".gz") else open(
            filepath, encoding="utf-8"
        ) as fIn:
            data = csv.reader(fIn, delimiter=self.delimiter, quoting=self.quoting)
            examples = []
            for id, row in enumerate(data):
                score = float(row[self.score_col_idx])
                if self.normalize_scores:  # Normalize to a 0...1 value
                    score = (score - self.min_score) / (self.max_score - self.min_score)

                s1 = row[self.s1_col_idx]
                s2 = row[self.s2_col_idx]
                examples.append(InputExample(guid=filename + str(id), texts=[s1, s2], label=score))

                if max_examples > 0 and len(examples) >= max_examples:
                    break

        return examples


class STSBenchmarkDataReader(STSDataReader):
    """
    Reader especially for the STS benchmark dataset. There, the sentences are in column 5 and 6, the score is in column 4.
    Scores are normalized from 0...5 to 0...1
    """

    def __init__(
        self,
        dataset_folder,
        s1_col_idx=5,
        s2_col_idx=6,
        score_col_idx=4,
        delimiter="\t",
        quoting=csv.QUOTE_NONE,
        normalize_scores=True,
        min_score=0,
        max_score=5,
    ):
        """
        Initializes the STSBenchmarkDataReader.

        Parameters
        ----------
        dataset_folder : str
            The folder containing the STS benchmark dataset files.

        s1_col_idx : int, optional
            Index of the column containing the first sentence. Default is 5.

        s2_col_idx : int, optional
            Index of the column containing the second sentence. Default is 6.

        score_col_idx : int, optional
            Index of the column containing the score. Default is 4.

        delimiter : str, optional
            The delimiter used in the dataset files. Default is "\t".

        quoting : int, optional
            The quoting style used in the csv files. Default is csv.QUOTE_NONE.

        normalize_scores : bool, optional
            Whether to normalize scores to a range of 0...1. Default is True.

        min_score : int, optional
            The minimum score in the dataset. Default is 0.
            
        max_score : int, optional
            The maximum score in the dataset. Default is 5.
        """
        super().__init__(
            dataset_folder=dataset_folder,
            s1_col_idx=s1_col_idx,
            s2_col_idx=s2_col_idx,
            score_col_idx=score_col_idx,
            delimiter=delimiter,
            quoting=quoting,
            normalize_scores=normalize_scores,
            min_score=min_score,
            max_score=max_score,
        )
