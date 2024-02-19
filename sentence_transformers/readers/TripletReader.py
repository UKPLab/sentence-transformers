from . import InputExample
import csv
import os


class TripletReader(object):
    """
    Reads in the a Triplet Dataset: Each line contains (at least) 3 columns, one anchor column (s1),
    one positive example (s2) and one negative example (s3)
    """

    def __init__(
        self,
        dataset_folder,
        s1_col_idx:int=0,
        s2_col_idx:int=1,
        s3_col_idx:int=2,
        has_header:bool=False,
        delimiter:str="\t",
        quoting=csv.QUOTE_NONE,
    ):
        """
        Initializes the TripletReader.

        Parameters
        ----------
        dataset_folder : str
            The folder containing the Triplet dataset files.

        s1_col_idx : int, optional
            Index of the column containing the anchor examples. Default is 0.

        s2_col_idx : int, optional
            Index of the column containing the positive examples. Default is 1.

        s3_col_idx : int, optional
            Index of the column containing the negative examples. Default is 2.

        has_header : bool, optional
            Whether the dataset files contain a header. Default is False.

        delimiter : str, optional
            The delimiter used in the dataset files. Default is "\t".

        quoting : int, optional
            The quoting style used in the csv files. Default is csv.QUOTE_NONE.
        """
        self.dataset_folder = dataset_folder
        self.s1_col_idx = s1_col_idx
        self.s2_col_idx = s2_col_idx
        self.s3_col_idx = s3_col_idx
        self.has_header = has_header
        self.delimiter = delimiter
        self.quoting = quoting

    def get_examples(self, filename:str, max_examples:int=0) -> list:
        """
        Reads examples from the Triplet dataset.

        Parameters
        ----------
        filename : str
            The name of the dataset file to read.

        max_examples : int, optional
            Maximum number of examples to read. Default is 0, meaning read all examples.

        Returns
        -------
        examples : list
            A list of InputExample objects representing the anchor, positive, and negative examples.
        """
        data = csv.reader(
            open(os.path.join(self.dataset_folder, filename), encoding="utf-8"),
            delimiter=self.delimiter,
            quoting=self.quoting,
        )
        examples = []
        if self.has_header:
            next(data)

        for id, row in enumerate(data):
            s1 = row[self.s1_col_idx]
            s2 = row[self.s2_col_idx]
            s3 = row[self.s3_col_idx]

            examples.append(InputExample(texts=[s1, s2, s3]))
            if max_examples > 0 and len(examples) >= max_examples:
                break

        return examples
