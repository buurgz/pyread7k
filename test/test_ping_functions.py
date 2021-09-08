import pytest
from dotenv import find_dotenv, load_dotenv
from pyread7k import FileDataset, Ping, PingDataset, PingType

from .conftest import bf_filepath, does_not_raise, filedataset

load_dotenv(find_dotenv())


def test_read_s7kfile(filedataset: FileDataset):
    assert isinstance(filedataset, FileDataset)


def test_filedataset_index_error(filedataset: FileDataset):
    with pytest.raises(IndexError):
        filedataset[1231241231]


def test_filedataaset_valid_index(filedataset: FileDataset):
    assert isinstance(filedataset[0], Ping)


@pytest.mark.parametrize("ping_number,default,expected,raises", [
    (123123123, 1, int, does_not_raise()),
    (123123123, "weird but works", str, does_not_raise()),
    (123123123, "weird but works", int, pytest.raises(AssertionError)),
])
class TestFiledatasetFunctions:
    """Grouped FileDataset tests"""
    @pytest.fixture
    def dataset(self):
        return FileDataset(bf_filepath, include=PingType.BEAMFORMED)

    def test_filedataset_get_by_number(self, ping_number, default, expected,
                                       raises, dataset):
        with raises:
            assert isinstance(dataset.get_by_number(ping_number, default),
                              expected)


def test_filedataset_compare_indexing_methods(filedataset: FileDataset):
    first_ping_by_index = filedataset[0]
    first_ping_by_number = filedataset.get_by_number(
        first_ping_by_index.sonar_settings.ping_number)
    assert first_ping_by_index == first_ping_by_number
