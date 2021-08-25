# %%
# import itertools
# import math
import datetime
import os
from test import bf_filepath

import psutil
import pytest
from numpy.testing import assert_almost_equal
from pyread7k import ConcatDataset, Dataset, PingDataset, PingType

# %%


def get_current_memory():
    """ Prints current memory of active process """
    pid = os.getpid()
    own_process = psutil.Process(pid)
    return own_process.memory_info()[0] / (1024**2)


@pytest.fixture
def dataset() -> PingDataset:
    return PingDataset(bf_filepath, include=PingType.BEAMFORMED)


@pytest.fixture
def ping(dataset):
    return dataset[10]


def test_sonar_settings_time(dataset):
    for p in dataset:
        assert isinstance(p.sonar_settings.frame.time, datetime.datetime)


def test_can_loop_multiple_times(dataset: PingDataset):
    loop1count = 0
    for p in dataset:
        p.position_set
        loop1count += 1

    loop2count = 0
    for p in dataset:
        p.position_set
        loop2count += 1

    assert loop1count == loop2count


def test_dataset_memory_use(dataset: PingDataset):
    init_memory = get_current_memory()
    for i, p in enumerate(dataset):
        # Make some ping calls to load data into the cached properties
        p.position_set
        p.roll_pitch_heave_set
        p.heading_set
        p.beamformed
        p.beam_geometry
        p.raw_iq

    # Check whether accessing all of the data of the pings resulted
    # in higher memeory use
    cur_memory = get_current_memory()
    assert cur_memory > init_memory

    # Minimize the ping data
    for p in dataset:
        p.minimize_memory()

    # Final check is whether the current memory is reduced
    # by minimizing the ping datastructure
    assert cur_memory > get_current_memory()


def test_new_dataset_class_read():
    ds = Dataset(bf_filepath)
    assert isinstance(ds, ConcatDataset)


def test_concatdataset_class_iterate():
    # Since dataset is a subclass of concat dataset we should be
    # able to iterate over the dataset and iterating multiple
    # times should also be possible
    ds = Dataset(bf_filepath, include=PingType.BEAMFORMED)
    a_old = None
    a_new = None
    for p in ds:
        a_old = a_new
        a_new = p.beamformed.amplitudes

    for p in ds:
        a_old = a_new
        a_new = p.beamformed.amplitudes


def test_concatdataset_iteration_items():
    # Since dataset is a subclass of concat dataset we should be
    # able to iterate over the dataset and iterating multiple
    # times should also be possible
    ds = Dataset(bf_filepath, include=PingType.BEAMFORMED)
    for i, p in enumerate(ds):
        assert p == ds[i]
        assert_almost_equal(p.beamformed.amplitudes,
                            ds[i].beamformed.amplitudes)
