from typing import Union, List


class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str = '', texts: List[str] = None, texts_tokenized: List[List[int]] = None, label: Union[int, float] = 0):
        """
        Creates one InputExample with the given texts, guid and label


        :param guid
            id for the example
        :param texts
            the texts for the example. Note, str.strip() is called on the texts
        :param texts_tokenized
            Optional: Texts that are already tokenized. If texts_tokenized is passed, texts must not be passed.
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = [text.strip() for text in texts] if texts is not None else texts
        self.texts_tokenized = texts_tokenized
        self.label = label

    def __str__(self):
        return "<InputExample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts))