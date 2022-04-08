# %%
from pyread7k import FileDataset, PingType

from .conftest import bf_filepath, ci_filepath


def test_read_beamformed():
    assert len(FileDataset(bf_filepath, include=PingType.BEAMFORMED)) > 0
