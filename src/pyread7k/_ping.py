"""
This module is an abstraction on top of the low-level 7k records, which allows
the user to work in terms of "pings" with associated data, instead of thinking
in the traditional 7k records.

Expected order of records for a ping:
7000, 7503, 1750, 7002, 7004, 7017, 7006, 7027, 7007, 7008, 7010, 7011, 7012,
7013, 7018, 7019, 7028, 7029, 7037, 7038, 7039, 7041, 7042, 7048, 7049, 7057,
7058, 7068, 7070
"""
# %%
from enum import Enum
from functools import cached_property
from typing import Optional, List
import math

import numpy as np

from ._utils import read_file_catalog, read_file_header, read_records, get_record_offsets
from . import _datarecord
from ._datarecord import DataParts


class LazyMap(dict):
    """
    An advanced defaultdict, where the initializer may depend on the key.
    """
    def __init__(self, initializer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initializer = initializer

    def __getitem__(self, key):
        if key not in self:
            self[key] = self.initializer(key)
        return super().__getitem__(key)


class Manager7k:
    """
    Internal class for Pings to share access to a file.
    """
    def __init__(self, fhandle, file_catalog):
        self.fhandle = fhandle
        self.file_catalog = file_catalog
        self._offsets_for_type = LazyMap(
            initializer=lambda key: get_record_offsets(
                key, self.file_catalog)
        )

    def get_next_record(self, record_type, offset_start, offset_end):
        """
        Get the offset and first record of type record_type which has a higher
        file offset than offset_start.
        """
        offset = self.get_next_offset(record_type, offset_start, offset_end)
        if offset is None:
            return None
        return self.read_record(record_type, offset)

    def get_next_offset(self, record_type, offset_start, offset_end):
        """
        Get the offset of type record_type which has a file offset between
        offset_start and offset_end.

        The data from a new ping always starts with a 7000 record, and so you
        can get the offset of a record for a specific ping by searching
        for an offset higher than the ping's 7000 record, but lower than the
        next ping's 7000 record.
        """
        record_offsets = self._offsets_for_type[record_type]
        next_index = np.searchsorted(record_offsets, offset_start, side="right")

        if next_index == len(record_offsets):
            # Reached end of offsets without match
            return None
        offset = record_offsets[next_index]
        if offset < offset_end:
            # No record exists in the interval
            return offset
        return None

    def read_record(self, record_type, offset):
        """
        Read a record from file using a known offset
        """
        self.fhandle.seek(offset)
        return _datarecord.record(record_type).read(self.fhandle)

    def get_records_during_ping(self, record_type, ping_start, ping_end, offset_hint):
        """
        Reads all records of record_type which are timestamped in the interval.

        Performs a brute-force search starting around the offset_hint. If the
        hint is good (which it should usually be), this is pretty efficient.

        Records of different types are not guaranteed to be chronological, so
        we cannot know a specific record interval to search.
        """
        record_offsets = self._offsets_for_type[record_type]
        initial_index = np.searchsorted(record_offsets, offset_hint)

        searching_backward = True
        searching_forward = True

        # Search forward in file
        forward_records = []
        index = initial_index
        while searching_forward:
            if index == len(record_offsets):
                # Reached end of file
                break
            next_record = self.read_record(record_type, record_offsets[index])
            if ping_end is not None and next_record.frame.time > ping_end:
                # Reached upper end of interval
                searching_forward = False
            elif not next_record.frame.time > ping_start:
                # Did not yet reach interval, backward search is unnecessary
                searching_backward = False
            else:
                forward_records.append(next_record)
            index += 1

        # Search backward in file
        backward_records = []
        index = initial_index - 1
        while searching_backward:
            if index == -1:
                break # Reached start of file

            next_record = self.read_record(record_type, record_offsets[index])
            if next_record.frame.time < ping_start:
                # Reached lower end of interval
                searching_backward = False
            elif ping_end is not None and not next_record.frame.time < ping_end:
                # Did not yet reach interval
                pass
            else:
                backward_records.append(next_record)
            index -= 1
        # Discovered in reverse order, so un-reverse
        backward_records.reverse()

        return backward_records + forward_records


class Ping:
    """
    A sound ping from a sonar, with associated data about settings and conditions.
    Properties of the ping are loaded efficiently on-demand.
    """

    minimizable_properties = ["beamformed", "tvg", "beam_geometry", "raw_iq"]

    def __init__(self, settings_record : DataParts, settings_offset : int,
                 next_record, next_offset : int, manager : Manager7k):

        # This is the only record always in-memory, as it defines the ping.
        self.sonar_settings : DataParts = settings_record
        self.ping_number : int = settings_record.header["ping_number"]

        self._manager = manager
        self._own_offset = settings_offset # This ping's start offset
        self._next_offset = next_offset # Next ping's start offset, meaning this ping has ended
        self.next_ping_start = (next_record.frame.time
            if next_record is not None else None)

        self._offset_map = LazyMap(
            initializer=lambda key: self._manager.get_next_offset(
                key, self._own_offset, self._next_offset)
        )

    def __str__(self) -> str:
        return "<Ping %i>" % self.sonar_settings.header["ping_number"]

    def minimize_memory(self) -> None:
        """
        Clears all memory-heavy properties.
        Retains offsets for easy reloading.
        """
        for key in self.minimizable_properties:
            if key in self.__dict__:
                del self.__dict__[key]

    def _get_single_associated_record(self, record_type : int):
        """
        Read a record associated with the ping. The requested record must:
        - Be the only of its type for the ping
        - Be located in the file between this ping's 7000 record and the next
          ping' 7000 record.
        """
        offset = self._offset_map[record_type]
        if offset is None:
            return None
        record = self._manager.read_record(record_type, offset)
        if "ping_number" in record.header:
            # If record contains ping number, we double-check validity
            assert record.header["ping_number"] == self.ping_number
        return record

    @cached_property
    def position_set(self) -> List[DataParts]:
        """ Returns all 1003 records timestamped within this ping. """
        return self._manager.get_records_during_ping(
            1003, self.sonar_settings.frame.time, self.next_ping_start,
            self._own_offset)

    @cached_property
    def roll_pitch_heave_set(self) -> List[DataParts]:
        """ Returns all 1012 records timestamped within this ping. """
        return self._manager.get_records_during_ping(
            1012, self.sonar_settings.frame.time, self.next_ping_start,
            self._own_offset)

    @cached_property
    def heading_set(self) -> List[DataParts]:
        """ Returns all 1013 records timestamped within this ping. """
        return self._manager.get_records_during_ping(
            1013, self.sonar_settings.frame.time, self.next_ping_start,
            self._own_offset)

    @cached_property
    def beam_geometry(self) -> Optional[DataParts]:
        """ Returns 7004 record """
        return self._get_single_associated_record(7004)

    @cached_property
    def tvg(self) -> Optional[DataParts]:
        """ Returns 7010 record """
        return self._get_single_associated_record(7010)

    @cached_property
    def has_beamformed(self) -> bool:
        """ Checks if the ping has 7018 data without reading it. """
        return self._offset_map[7018] is not None

    @cached_property
    def beamformed(self) -> Optional[DataParts]:
        """ Returns 7018 record """
        return self._get_single_associated_record(7018)

    @cached_property
    def has_raw_iq(self) -> bool:
        """ Checks if the ping has 7038 data without reading it. """
        return self._offset_map[7038] is not None

    @cached_property
    def raw_iq(self) -> Optional[DataParts]:
        """ Returns 7038 record """
        return self._get_single_associated_record(7038)


# %%
class PingType(Enum):
    """ Kinds of pings based on what data they have available """
    BEAMFORMED = 1
    IQ = 2
    ANY = 3


class PingDataset:
    """
    Indexable PyTorch dataset returning Pings from a 7k file.

    Provides random access into pings in a file with minimal overhead.
    """
    def __init__(self, filename, include : PingType=PingType.ANY):
        """
        if include argument is not ANY, pings will be filtered.
        """
        self.filename = filename

        self.fhandle = open(filename, "rb", buffering=0)

        file_header = read_file_header(self.fhandle)
        file_catalog = read_file_catalog(self.fhandle, file_header)

        manager = Manager7k(self.fhandle, file_catalog)

        settings_records = read_records(7000, self.fhandle, file_catalog)
        settings_offsets = get_record_offsets(7000, file_catalog)
        settings_and_offsets = list(zip(settings_records, settings_offsets))

        pings = [Ping(rec, offset, next_rec, next_off, manager) for (rec, offset), (next_rec, next_off)
                          in zip(settings_and_offsets,
                                 settings_and_offsets[1:] + [(None, math.inf),])]

        if include == PingType.BEAMFORMED:
            self.pings = [p for p in pings if p.has_beamformed]
        elif include == PingType.IQ:
            self.pings = [p for p in pings if p.has_raw_iq]
        elif include == PingType.ANY:
            self.pings = pings
        else:
            raise NotImplementedError("Encountered unknown PingType: %s"  % str(include))

    def __len__(self) -> int:
        return len(self.pings)

    def __getitem__(self, index: int) -> Ping:
        return self.pings[index]


class ConcatDataset:
    """
    Reimplementation of Pytorch ConcatDataset to avoid dependency
    """
    def __init__(self, datasets):

        self.cum_lengths = np.cumsum([len(d) for d in datasets])
        self.datasets = datasets

    def __len__(self):
        return self.cum_lengths[-1]

    def __getitem__(self, index):
        if index < 0:
            if -index > len(self):
                raise ValueError("Index out of range")
            index = len(self) + index
        dataset_index = np.searchsorted(self.cum_lengths, index, side="right")
        if dataset_index == 0:
            sample_index = index
        else:
            sample_index = index - self.cum_lengths[dataset_index - 1]
        return self.datasets[dataset_index][sample_index]
