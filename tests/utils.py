from __future__ import annotations

import tempfile


class SafeTemporaryDirectory(tempfile.TemporaryDirectory):
    """
    The GitHub Actions CI on Windows sometimes raises a NotADirectoryError when cleaning up the temporary directory.
    This class is a workaround to avoid the error.

    Unlike tempfile.TemporaryDirectory(ignore_cleanup_errors=True), this also works on Python 3.8 and 3.9.
    """

    def __exit__(self, *args, **kwargs):
        try:
            super().__exit__(*args, **kwargs)
        except NotADirectoryError:
            pass
