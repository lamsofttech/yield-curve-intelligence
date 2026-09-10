"""Shared pytest fixtures/helpers."""

import warnings

import pytest


@pytest.fixture
def warnings_as_errors():
    """Context manager fixture: promotes UserWarning to an exception inside
    the `with` block, so a test can assert that a specific code path does
    *not* warn (rather than only checking that it doesn't raise).
    """

    class _Guard:
        def __enter__(self):
            warnings.simplefilter("error", UserWarning)
            return self

        def __exit__(self, *exc):
            warnings.resetwarnings()

    return _Guard()
