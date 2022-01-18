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


class TestFiledatasetFunctions:
    """Grouped FileDataset tests"""
    @pytest.fixture
    def dataset(self):
        return FileDataset(bf_filepath, include=PingType.BEAMFORMED)

    @pytest.mark.parametrize("ping_number,default,expected,raises", [
        (123123123, 1, int, does_not_raise()),
        (123123123, "weird but works", str, does_not_raise()),
        (123123123, "weird but works", int, pytest.raises(AssertionError)),
        ("not a ping", None, int, pytest.raises(TypeError)),
    ])
    def test_filedataset_get_by_number_input_handling(self, ping_number, default, expected,
                                       raises, dataset):
        with raises:
            assert isinstance(dataset.get_by_number(ping_number, default),
                              expected)

    def test_filedataset_get_by_number(self, dataset):
        ping_numbers = dataset.ping_numbers
        for pn in ping_numbers:
            assert isinstance(dataset.get_by_number(pn), Ping)

    def test_filedataset_index_of(self, dataset):
        for p in dataset:
            ping_number = int(p.sonar_settings.ping_number)
            assert isinstance(dataset.index_of(ping_number), int)

