from __future__ import annotations

import os
import tempfile

from huggingface_hub import get_hf_file_metadata as hf_hub_get_hf_file_metadata
from transformers.utils.hub import http_user_agent


def is_ci() -> bool:
    """
    Check if the code is running in a Continuous Integration (CI) environment.
    This is determined by checking for the presence of certain environment variables.
    """
    return "GITHUB_ACTIONS" in os.environ


class SafeTemporaryDirectory(tempfile.TemporaryDirectory):
    """
    The GitHub Actions CI on Windows sometimes raises a NotADirectoryError when cleaning up the temporary directory.
    This class is a workaround to avoid the error.

    Unlike tempfile.TemporaryDirectory(ignore_cleanup_errors=True), this also works on Python 3.9.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs["ignore_cleanup_errors"] = True
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            del kwargs["ignore_cleanup_errors"]
            super().__init__(*args, **kwargs)

    def __exit__(self, *args, **kwargs):
        try:
            super().__exit__(*args, **kwargs)
        except NotADirectoryError:
            pass


def get_hf_file_metadata_with_user_agent(*args, user_agent: dict | str | None = None, **kwargs):
    """
    This function is a wrapper around `huggingface_hub.get_hf_file_metadata` that defaults to using the user agent
    from `transformers.utils.hub.http_user_agent()` if no user agent is provided.

    This is intended to help prevent 429 errors when using the `huggingface_hub` library in CI environments.
    """
    return hf_hub_get_hf_file_metadata(*args, user_agent=user_agent or http_user_agent(), **kwargs)
