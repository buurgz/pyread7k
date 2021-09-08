# %%
from pyread7k import FileDataset, PingType

from .conftest import bf_filepath, iq_filepath


def test_read_beamformed():
    assert (len(FileDataset(bf_filepath, include=PingType.BEAMFORMED)) > 0)


def test_read_iq():
    assert len(FileDataset(iq_filepath, include=PingType.IQ)) > 0
