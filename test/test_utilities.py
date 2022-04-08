# %%
import datetime
import os
from collections.abc import Iterable

import psutil
from pyread7k import (
    FileDataset,
)
from pyread7k._utils import build_file_catalog


from .conftest import bf_filepath


def test_file_catalog_validity():
    # Build a catalog
    with open(bf_filepath, "rb", 0) as fhandle:
        fc_c = build_file_catalog(fhandle)

    # Get a catalog
    fc_e = FileDataset(bf_filepath)[0]._reader.file_catalog

    assert fc_c.size == fc_e.size
