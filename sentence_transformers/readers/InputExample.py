from typing import Union, List


class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """

    def __init__(self, guid: str = "", texts: List[str] = None, label: Union[int, float] = 0):
        """
        Creates one InputExample with the given texts, guid and label

        Parameters
        ----------
        guid : str, optional
            Id for the example. Default is an empty string.

        texts : List[str], optional
            The texts for the example. Default is None.

        label : Union[int, float], optional
            The label for the example. Default is 0.
        """
        self.guid = guid
        self.texts = texts
        self.label = label

    def __str__(self):
        """
        String representation of the InputExample.

        Parameters
        ----------
        None

        Returns
        -------
        str
            String representation of the InputExample.
        """
        return "<InputExample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts))
