from . import InputExample
import os


class LabelSentenceReader:
    """Reads in a file that has at least two columns: a label and a sentence.
    This reader can for example be used with the BatchHardTripletLoss.
    Maps labels automatically to integers"""

    def __init__(self, folder, label_col_idx:int=0, sentence_col_idx:int=1, separator:str="\t"):
        """
        Initializes the LabelSentenceReader.

        Parameters
        ----------
        folder : str
            The folder containing the files to be read.

        label_col_idx : int, optional
            The index of the column containing labels in the file. Default is 0.

        sentence_col_idx : int, optional
            The index of the column containing sentences in the file. Default is 1.

        separator : str, optional
            The separator used in the file. Default is "\t".
        """
        self.folder = folder
        self.label_map = {}
        self.label_col_idx = label_col_idx
        self.sentence_col_idx = sentence_col_idx
        self.separator = separator

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
        examples = []

        id = 0
        for line in open(os.path.join(self.folder, filename), encoding="utf-8"):
            splits = line.strip().split(self.separator)
            label = splits[self.label_col_idx]
            sentence = splits[self.sentence_col_idx]

            if label not in self.label_map:
                self.label_map[label] = len(self.label_map)

            label_id = self.label_map[label]
            guid = "%s-%d" % (filename, id)
            id += 1
            examples.append(InputExample(guid=guid, texts=[sentence], label=label_id))

            if 0 < max_examples <= id:
                break

        return examples
