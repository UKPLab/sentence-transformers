from . import InputExample
import gzip
import os


class NLIDataReader(object):
    """
    Reads in the Stanford NLI dataset and the MultiGenre NLI dataset
    """

    def __init__(self, dataset_folder:str):
        """
        Initializes the NLIDataReader.

        Parameters
        ----------
        dataset_folder : str
            The folder containing the dataset files.
        """
        self.dataset_folder = dataset_folder

    def get_examples(self, filename:str, max_examples:int=0) -> list:
        """
        Reads examples from a file.

        Parameters
        ----------
        filename : str
            The name of the file to read.

        max_examples : int, optional
            Maximum number of examples to read. Default is 0, meaning read all examples.

        Returns
        -------
        examples : list
            A list of InputExample objects.
        """
        s1 = gzip.open(os.path.join(self.dataset_folder, "s1." + filename), mode="rt", encoding="utf-8").readlines()
        s2 = gzip.open(os.path.join(self.dataset_folder, "s2." + filename), mode="rt", encoding="utf-8").readlines()
        labels = gzip.open(
            os.path.join(self.dataset_folder, "labels." + filename), mode="rt", encoding="utf-8"
        ).readlines()

        examples = []
        id = 0
        for sentence_a, sentence_b, label in zip(s1, s2, labels):
            guid = "%s-%d" % (filename, id)
            id += 1
            examples.append(InputExample(guid=guid, texts=[sentence_a, sentence_b], label=self.map_label(label)))

            if 0 < max_examples <= len(examples):
                break

        return examples

    @staticmethod
    def get_labels():
        """
        Returns the labels and their corresponding integer mappings.

        Parameters
        ----------
        None

        Returns
        -------
        labels : dict
            A dictionary containing label names as keys and their integer mappings as values.
        """
        return {"contradiction": 0, "entailment": 1, "neutral": 2}

    def get_num_labels(self) -> int:
        """
        Returns the number of unique labels.

        Returns
        -------
        num_labels : int
            The number of unique labels.
        """
        return len(self.get_labels())

    def map_label(self, label:str) -> int:
        """
        Maps a label to its corresponding integer mapping.

        Parameters
        ----------
        label : str
            The label to map.

        Returns
        -------
        label_id : int
            The integer mapping of the label.
        """
        return self.get_labels()[label.strip().lower()]
