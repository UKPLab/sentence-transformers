from typing import List, Union


class InputExample:
    """Structure for one input example with texts, the label and a unique id"""

    def __init__(self, guid: str = "", texts: List[str] = None, label: Union[int, float] = 0):
        """
        Creates one InputExample with the given texts, guid and label

        Args:
            guid: id for the example
            texts: the texts for the example.
            label: the label for the example
        """
        self.guid = guid
        self.texts = texts
        self.label = label

    def __str__(self):
        return "<InputExample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts))
