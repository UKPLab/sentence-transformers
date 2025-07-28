from __future__ import annotations

import csv
import importlib
import logging
from contextlib import contextmanager


def fullname(o) -> str:
    """
    Gives a full name (package_name.class_name) for a class / object in Python. Will
    be used to load the correct classes from JSON files

    Args:
        o: The object for which to get the full name.

    Returns:
        str: The full name of the object.

    Example:
        >>> from sentence_transformers.losses import MultipleNegativesRankingLoss
        >>> from sentence_transformers import SentenceTransformer
        >>> from sentence_transformers.util import fullname
        >>> model = SentenceTransformer('all-MiniLM-L6-v2')
        >>> loss = MultipleNegativesRankingLoss(model)
        >>> fullname(loss)
        'sentence_transformers.losses.MultipleNegativesRankingLoss.MultipleNegativesRankingLoss'
    """

    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__  # Avoid reporting __builtin__
    else:
        return module + "." + o.__class__.__name__


def import_from_string(dotted_path: str) -> type:
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.

    Args:
        dotted_path (str): The dotted module path.

    Returns:
        Any: The attribute/class designated by the last name in the path.

    Raises:
        ImportError: If the import failed.

    Example:
        >>> import_from_string('sentence_transformers.losses.MultipleNegativesRankingLoss')
        <class 'sentence_transformers.losses.MultipleNegativesRankingLoss.MultipleNegativesRankingLoss'>
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError:
        msg = f"{dotted_path} doesn't look like a module path"
        raise ImportError(msg)

    try:
        module = importlib.import_module(dotted_path)
    except Exception:
        module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = f'Module "{module_path}" does not define a "{class_name}" attribute/class'
        raise ImportError(msg)


@contextmanager
def disable_datasets_caching():
    """
    A context manager that will disable caching in the datasets library.
    """
    from datasets import disable_caching, enable_caching, is_caching_enabled

    is_originally_enabled = is_caching_enabled()

    try:
        if is_originally_enabled:
            disable_caching()
        yield
    finally:
        if is_originally_enabled:
            enable_caching()


@contextmanager
def disable_logging(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.

    Args:
        highest_level: the maximum logging level allowed.
    """
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)


def append_to_last_row(csv_path, additional_data):
    # Read the entire CSV file
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) > 1:  # Make sure there's at least one data row (after the header)
        # Append the additional data to the last row
        rows[-1].extend(additional_data)

        # Write the entire file back
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        return True
    return False
