# pylint: skip-file
"""Shared test fixtures

"""
import os
from contextlib import contextmanager

import pytest
from pyread7k import (ConcatDataset, FileDataset, FolderDataset, Ping,
                      PingDataset, PingType)

from .context import bf_filepath, iq_filepath, root_dir


@pytest.fixture
def filedataset() -> FileDataset:
    return FileDataset(bf_filepath, include=PingType.BEAMFORMED)


@pytest.fixture
def ping(dataset) -> Ping:
    return dataset[10]


@pytest.fixture
def folderdataset(goodfile) -> FolderDataset:
    return FolderDataset(os.path.dirname(goodfile),
                         include=PingType.BEAMFORMED)


@contextmanager
def does_not_raise():
    yield
