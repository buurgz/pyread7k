# %%
# import itertools
# import math
import datetime
import os
from collections.abc import Iterable

import psutil
from pyread7k.records import DetectionFlags, RemoteControlSonarSettings
import pytest
from numpy.testing import assert_almost_equal
from pyread7k import (
    ConcatDataset,
    FileDataset,
    FolderDataset,
    PingDataset,
    PingType,
    S7KRecordReader,
)


from .conftest import bf_filepath, does_not_raise, filedataset


def get_current_memory() -> float:
    """ Prints current memory of active process """
    pid = os.getpid()
    own_process = psutil.Process(pid)
    return own_process.memory_info()[0] / (1024 ** 2)


def test_sonar_settings_time(filedataset):
    for p in filedataset:
        assert isinstance(p.sonar_settings.frame.time, datetime.datetime)


def test_can_loop_multiple_times(filedataset: FileDataset):
    loop1count = 0
    for p in filedataset:
        p.position_set
        loop1count += 1

    loop2count = 0
    for p in filedataset:
        p.position_set
        loop2count += 1

    assert loop1count == loop2count


def test_dataset_memory_use(filedataset: FileDataset):
    init_memory = get_current_memory()
    for i, p in enumerate(filedataset):
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
    for p in filedataset:
        p.minimize_memory()

    # Final check is whether the current memory is reduced
    # by minimizing the ping datastructure
    assert cur_memory > get_current_memory()


def test_new_dataset_class_read():
    ds = FolderDataset(os.path.dirname(bf_filepath))
    assert isinstance(ds, ConcatDataset)


def test_concatdataset_class_iterate():
    # Since dataset is a subclass of concat dataset we should be
    # able to iterate over the dataset and iterating multiple
    # times should also be possible
    ds = ConcatDataset([FileDataset(bf_filepath, include=PingType.BEAMFORMED)])
    a_old = None
    a_new = None
    for p in ds:
        a_old = a_new
        a_new = p.beamformed.amplitudes

    for p in ds:
        a_old = a_new
        a_new = p.beamformed.amplitudes


def test_dataset_iter():
    ds = FolderDataset(os.path.dirname(bf_filepath), include=PingType.BEAMFORMED)
    dsiter = iter(ds)
    assert isinstance(dsiter, Iterable)


def test_folder_dataset_iterate():
    # Since dataset is a subclass of concat dataset we should be
    # able to iterate over the dataset and iterating multiple
    # times should also be possible
    ds = FolderDataset(os.path.dirname(bf_filepath), include=PingType.BEAMFORMED)
    for i, p in enumerate(ds):
        if i % 10 != 0:
            # For speed, only sample every 10th ping
            continue
        assert p == ds[i]
        assert_almost_equal(p.beamformed.amplitudes, ds[i].beamformed.amplitudes)


def test_folder_dataset_is_sorted():
    ds = FolderDataset(os.path.dirname(bf_filepath), include=PingType.BEAMFORMED)
    old_timestamp = -10000
    timestamp = -10000
    for p in ds:
        timestamp = p.sonar_settings.frame.time.timestamp()
        assert timestamp > old_timestamp
        old_timestamp = timestamp


@pytest.mark.parametrize(
    "folderpath,expected,raises",
    [
        ("this is not a real filepath", None, pytest.raises(FileNotFoundError)),
        (os.path.join(".", "test"), None, pytest.raises(ValueError)),
        (123, None, pytest.raises(TypeError)),
        (os.path.dirname(bf_filepath), FolderDataset, does_not_raise()),
        (os.path.dirname(bf_filepath), ConcatDataset, does_not_raise()),
    ],
)
def test_folderdataset_input(folderpath, expected, raises):
    with raises:
        result = FolderDataset(folderpath, include=PingType.BEAMFORMED)
        assert isinstance(result, expected)


def test_records_reader_linear_read():
    stream_reader = S7KRecordReader(
        bf_filepath,
    )
    records = [record for record in stream_reader]
    assert len(records) > 0


def test_records_reader_bad_path_exception():
    with pytest.raises(FileNotFoundError):
        [r for r in S7KRecordReader("not a real path")]


def test_records_reader_non_existing_record_returns_empty_recordlist():
    stream_reader = S7KRecordReader(bf_filepath, records_to_read=[1202391283098123])
    records = [record for record in stream_reader]
    assert len(records) == 0


class Test7027Records:
    def test_detection_shape(self, filedataset):
        for ping in filedataset:
            if ping.raw_detections is None:
                continue
            assert (
                ping.raw_detections.detections.shape[0]
                == ping.raw_detections.detection_count
            )

    def test_detection_flag_type(self, filedataset):
        for ping in filedataset:
            if ping.raw_detections is None:
                continue
            assert isinstance(
                ping.raw_detections.parse_detection_flags(
                    ping.raw_detections.detections[0]
                )[0],
                DetectionFlags,
            )

    def test_columns_in_detections(self, filedataset):
        for ping in filedataset:
            if ping.raw_detections is None:
                continue
            for name in [
                "beam_descriptor",
                "detection_point",
                "rx_angle",
                "flags",
                "quality",
                "uncertainty",
                "intensity",
            ]:
                # The usual test file does not have min or max limit, but is still valid
                assert name in ping.raw_detections.detections.dtype.names

    def test_file_has_7027(self, filedataset):
        has_raw_detections = False
        for ping in filedataset:
            if ping.raw_detections is not None:
                has_raw_detections = True

        assert has_raw_detections


class TestRecord7503:
    def test_has_record(self, filedataset):
        assert hasattr(filedataset[0], "remote_control_sonar_settings")

    def test_record_has_attributes(self, filedataset):
        ping = filedataset[0]
        for field in RemoteControlSonarSettings.__annotations__.keys():
            assert hasattr(ping.remote_control_sonar_settings, field)

    def test_record_attributes_have_data(self, filedataset):
        ping = filedataset[0]
        for field in RemoteControlSonarSettings.__annotations__.keys():
            assert getattr(ping.remote_control_sonar_settings, field, None) is not None
