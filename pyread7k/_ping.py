"""
This module is an abstraction on top of the low-level 7k records, which allows
the user to work in terms of "pings" with associated data, instead of thinking
in the traditional 7k records.

Expected order of records for a ping:
7000, 7503, 1750, 7002, 7004, 7017, 7006, 7027, 7007, 7008, 7010, 7011, 7012,
7013, 7018, 7019, 7028, 7029, 7037, 7038, 7039, 7041, 7042, 7048, 7049, 7057,
7058, 7068, 7070

"""
import bisect
import logging
import os
import sys
import warnings
from abc import ABCMeta, abstractmethod, abstractproperty
from datetime import datetime, timedelta
from enum import Enum
from io import BytesIO
from itertools import chain
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast
)

import geopy
import numpy as np

from . import _datarecord, records
from ._utils import build_file_catalog, cached_property, window

logger = logging.getLogger(__name__)


class PingType(Enum):
    """ Kinds of pings based on what data they have available """

    BEAMFORMED = 1
    IQ = 2
    ANY = 3


class CatalogIssueHandling(Enum):
    """How to handle file catalog issues.
    It has 4 modes, which handle catalog issues differently.

    The first is to handle corrupt catalogs. This makes sure that
    parsing of S7k files using pyread7ks various file and folder datasets is
    smooth, even when the file catalog is corrupt, it can construct the file
    catalog with an initial linear read.

    The second is to handle corrupt file catalogs but warn the user
    that the file catalog is corrupt. 

    The third is to raise if the file catalog is missing or corrupt

    """

    HANDLE_CORRUPT = 1
    HANDLE_BUT_WARN = 2
    RAISE = 3


class S7KReader(metaclass=ABCMeta):
    """
    Base abstract class of S7K readers

    *Note*: The current S7KReader API is considered unstable and may change in the future.
    """

    @abstractproperty
    def catalog_issue_handling(self) -> CatalogIssueHandling:
        """Return the catalog issue handling property."""

    @cached_property
    def file_header(self) -> records.FileHeader:
        """Return the file header record for this reader"""
        fileheader = cast(records.FileHeader, self._read_record(7200, 0))
        return fileheader

    @cached_property
    def file_catalog(self) -> records.FileCatalog:
        """Return the file catalog record for this reader"""
        try:
            if self.file_header.catalog_offset == 0:
                raise _datarecord.CorruptFileCatalog
            filecatalog = cast(
                records.FileCatalog,
                self._read_record(7300, self.file_header.catalog_offset),
            )
        except _datarecord.CorruptFileCatalog as exc:
            if self._catalog_issue_handling == CatalogIssueHandling.RAISE:
                raise exc
            elif (
                self._catalog_issue_handling == CatalogIssueHandling.HANDLE_BUT_WARN
            ):
                logger.warning(
                    "File catalog was corrupt but a new one was generated."
                )
            filecatalog = self._build_file_catalog()
        except Exception as exc:
            raise exc
        return filecatalog

    @cached_property
    def configuration(self) -> records.Configuration:
        """Return the configuration record for this reader"""
        offsets = self._get_offsets(7001)
        assert len(offsets) == 1
        return cast(records.Configuration, self._read_record(7001, offsets[0]))

    def iter_pings(self, include: PingType = PingType.ANY) -> Iterator["Ping"]:
        """Iterate over Pings. if include argument is not ANY, filter pings by type"""
        settings_records = chain(self._iter_offset_records(7000), [None])
        pings = (
            Ping(
                cast(Tuple[int, records.SonarSettings], offset_record),
                cast(Optional[Tuple[int, records.SonarSettings]], next_offset_record),
                reader=self,
            )
            for offset_record, next_offset_record in window(settings_records, 2)
        )
        if include == PingType.ANY:
            return pings
        if include == PingType.BEAMFORMED:
            return (p for p in pings if p.has_beamformed)
        if include == PingType.IQ:
            return (p for p in pings if p.has_raw_iq)
        raise NotImplementedError(f"Encountered unknown PingType: {include!r}")

    def get_first_offset(
        self, record_type: int, offset_start: int, offset_end: int
    ) -> Optional[Tuple[int, int]]:
        """
        Get the offset of the first record of type record_type which has a
        file offset between offset_start and offset_end.
        """
        offsets = self._get_offsets(record_type)
        i = bisect.bisect_right(offsets, (offset_start, 0))
        if i < len(offsets) and offsets[i][0] < offset_end:
            return offsets[i]

    def read_first_record(
        self, record_type: int, offset_start: int, offset_end: int
    ) -> Optional[records.BaseRecord]:
        """
        Read the first record of type record_type which has a file offset between
        offset_start and offset_end.
        """
        offset = self.get_first_offset(record_type, offset_start, offset_end)
        return self._read_record(record_type, offset[0], offset[1]) if offset is not None else None

    def read_records_during_ping(
        self,
        record_type: int,
        ping_start: datetime,
        ping_end: datetime,
        offset_hint: int,
    ) -> List[records.BaseRecord]:
        """
        Read all records of record_type which are timestamped in the interval between
        ping_start and ping_end. An offset_hint is given as an initial offset of a record
        close to the interval, to be used if it can make the search more efficient.
        """
        # Performs a brute-force search starting around the offset_hint. If the
        # hint is good (which it should usually be), this is pretty efficient.
        #
        # Records of different types are not guaranteed to be chronological, so
        # we cannot know a specific record interval to search.
        offsets = self._get_offsets(record_type)
        initial_index = bisect.bisect_left(offsets, (offset_hint, 0))

        # Search forward in file
        forward_records = []
        searching_backward = True
        for index in range(initial_index, len(offsets)):
            next_record = self._read_record(record_type, *offsets[index])
            next_record_time = next_record.frame.time
            if next_record_time > ping_end:
                # Reached upper end of interval
                break
            elif next_record_time <= ping_start:
                # Did not yet reach interval, backward search is unnecessary
                searching_backward = False
            else:
                forward_records.append(next_record)

        if not searching_backward:
            return forward_records

        # Search backward in file
        backward_records = []
        for index in range(initial_index - 1, -1, -1):
            next_record = self._read_record(record_type, *offsets[index])
            next_record_time = next_record.frame.time
            if next_record_time < ping_start:
                # Reached lower end of interval
                break
            elif next_record_time >= ping_end:
                # Did not yet reach interval
                pass
            else:
                backward_records.append(next_record)

        # Discovered in reverse order, so un-reverse
        backward_records.reverse()
        backward_records.extend(forward_records)
        return backward_records

    def _read_record(self, record_type: int, offset: int, size: Optional[int]=None) -> records.BaseRecord:
        """Read a record of record_type at the given offset"""
        if size is not None:
            # If we have size, we can avoid some file operations by reading
            # into buffer before parsing record.
            bytes_wrapper = BytesIO(self._get_stream_for_read(offset).read(size))
            return _datarecord.record(record_type).read(bytes_wrapper)
        return _datarecord.record(record_type).read(self._get_stream_for_read(offset))

    def _iter_offset_records(
        self, record_type: int
    ) -> Iterator[Tuple[int, records.BaseRecord]]:
        """Generate all the (offset, record) tuples for the given record type"""
        for offset, size in self._get_offsets(record_type):
            yield offset, self._read_record(record_type, offset, size)

    def _get_offsets(self, record_type: int) -> Sequence[Tuple[int, int]]:
        """
        Return all (offset, size) tuples for the given record type.
        Does not handle 7300, since it is not in the file catalog.
        """
        try:
            return self.__cached_offsets[record_type]
        except (AttributeError, KeyError) as ex:
            offsets = [
                (offset, size)
                for offset, rt, size in zip(self.file_catalog.offsets,
                    self.file_catalog.record_types, self.file_catalog.sizes)
                if rt == record_type
            ]

            if isinstance(ex, AttributeError):
                self.__cached_offsets: Dict[int, Sequence[int]] = {}
            return self.__cached_offsets.setdefault(record_type, offsets)

    @abstractmethod
    def _get_stream_for_read(self, offset: int) -> BinaryIO:
        """Return a byte stream for reading a record at the given offset"""

    @abstractmethod
    def _build_file_catalog(self) -> records.FileCatalog:
        """ Build the file catalog. """


class S7KFileReader(S7KReader):
    """Reader class for s7k files"""

    def __init__(
        self,
        file: Union[str, bytes, os.PathLike, BinaryIO],
        catalog_issue_handling: CatalogIssueHandling = CatalogIssueHandling.RAISE,
    ):
        if isinstance(file, str) or isinstance(file, bytes) or isinstance(file, os.PathLike):
            self._fhandle = open(file, "rb", 0)
        else:
            self._fhandle = file
            file.seek(0)
        self._catalog_issue_handling = catalog_issue_handling

    def _get_stream_for_read(self, offset: int) -> BinaryIO:
        self._fhandle.seek(offset)
        return self._fhandle

    def _build_file_catalog(self) -> records.FileCatalog:
        self._fhandle.seek(0)
        return build_file_catalog(self._fhandle)

    def __getstate__(self) -> Dict[str, Any]:
        """ Remove unpicklable file handle from dict before pickling. """
        state = self.__dict__.copy()
        del state["_fhandle"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """ Open new file handle after unpickling. """
        self.__dict__.update(state)
        self._fhandle = open(self._filename, "rb", buffering=0)

    def __del__(self) -> None:
        self._fhandle.close()

    @property
    def catalog_issue_handling(self) -> CatalogIssueHandling:
        """Return the catalog issue handling property."""
        return self._catalog_issue_handling



class Ping:
    """
    A sound ping from a sonar, with associated data about settings and conditions.
    Properties of the ping are loaded efficiently on-demand.
    """

    def __init__(
        self,
        offset_record: Tuple[int, records.SonarSettings],
        next_offset_record: Optional[Tuple[int, records.SonarSettings]],
        reader: S7KReader,
    ):
        self._reader = reader
        self._offset, self.sonar_settings = offset_record

        if next_offset_record is not None:
            self._next_offset, next_sonar_settings = next_offset_record
            self._next_ping_start = next_sonar_settings.frame.time
        else:
            self._next_offset = sys.maxsize
            self._next_ping_start = datetime.max

    def __str__(self) -> str:
        return f"<Ping {self.ping_number}>"

    @property
    def ping_number(self) -> int:
        """Return ping number from the sonar settings record."""
        return self.sonar_settings.ping_number

    @property
    def configuration(self) -> records.Configuration:
        """Return the 7001 record, which is shared for all pings in a file"""
        return self._reader.configuration

    @cached_property
    def position_set(self) -> List[records.Position]:
        """ Returns all 1003 records timestamped within this ping. """
        return cast(List[records.Position], self._read_records(1003))

    @cached_property
    def depth_set(self) -> List[records.Depth]:
        """ Returns all 1008 records timestamped within this ping. """
        return cast(List[records.Depth], self._read_records(1008))

    @cached_property
    def roll_pitch_heave_set(self) -> List[records.RollPitchHeave]:
        """ Returns all 1012 records timestamped within this ping. """
        return cast(List[records.RollPitchHeave], self._read_records(1012))

    @cached_property
    def heading_set(self) -> List[records.Heading]:
        """ Returns all 1013 records timestamped within this ping. """
        return cast(List[records.Heading], self._read_records(1013))

    @cached_property
    def pan_tilt_roll_set(self) -> List[records.PanTiltRoll]:
        """ Returns all 1017 records timestamped within this ping. """
        return cast(List[records.PanTiltRoll], self._read_records(1017))

    @cached_property
    def velocity_set(self) -> List[records.Velocity]:
        """ Returns all 1018 records timestamped within this ping. """
        return cast(List[records.Velocity], self._read_records(1018))

    @cached_property
    def beam_geometry(self) -> Optional[records.BeamGeometry]:
        """ Returns 7004 record """
        return cast(Optional[records.BeamGeometry], self._read_record(7004))

    @cached_property
    def tvg(self) -> Optional[records.TVG]:
        """ Returns 7010 record """
        return cast(Optional[records.TVG], self._read_record(7010))

    @cached_property
    def has_beamformed(self) -> bool:
        """ Checks if the ping has 7018 data without reading it. """
        return (
            self._reader.get_first_offset(7018, self._offset, self._next_offset)
            is not None
        )

    @cached_property
    def beamformed(self) -> Optional[records.Beamformed]:
        """ Returns 7018 record """
        return cast(Optional[records.Beamformed], self._read_record(7018))

    @cached_property
    def raw_detections(self) -> Optional[records.RawDetectionData]:
        """ Returns 7027 record """
        return cast(Optional[records.RawDetectionData], self._read_record(7027))

    @cached_property
    def snippets(self) -> Optional[records.SnippetData]:
        """ Returns 7028 record """
        return cast(Optional[records.SnippetData], self._read_record(7028))

    @cached_property
    def has_raw_iq(self) -> bool:
        """ Checks if the ping has 7038 data without reading it. """
        return (
            self._reader.get_first_offset(7038, self._offset, self._next_offset)
            is not None
        )

    @cached_property
    def raw_iq(self) -> Optional[records.RawIQ]:
        """ Returns 7038 record """
        return cast(Optional[records.RawIQ], self._read_record(7038))

    @cached_property
    def gps_position(self) -> geopy.Point:
        lat = self.position_set[0].latitude * 180 / np.pi
        long = self.position_set[0].longitude * 180 / np.pi
        return geopy.Point(lat, long)

    def receiver_motion_for_sample(
        self, sample: int
    ) -> Tuple[records.RollPitchHeave, records.Heading]:
        """ Find the most appropriate motion data for a sample based on time """
        time = self.sonar_settings.frame.time + timedelta(
            seconds=sample / self.sonar_settings.sample_rate
        )
        rph_index = min(
            bisect.bisect_left([m.frame.time for m in self.roll_pitch_heave_set], time),
            len(self.roll_pitch_heave_set) - 1,
        )
        heading_index = min(
            bisect.bisect_left([m.frame.time for m in self.heading_set], time),
            len(self.heading_set) - 1,
        )
        return self.roll_pitch_heave_set[rph_index], self.heading_set[heading_index]

    def minimize_memory(self) -> None:
        """
        Clears all memory-heavy properties.
        Retains offsets for easy reloading.
        """
        for key in "beamformed", "tvg", "beam_geometry", "raw_iq", "snippets", "raw_detections":
            if key in self.__dict__:
                del self.__dict__[key]

    def _read_record(self, record_type: int) -> Optional[records.BaseRecord]:
        record = self._reader.read_first_record(
            record_type, self._offset, self._next_offset
        )
        if record is not None:
            ping_number = self.ping_number
            assert getattr(record, "ping_number", ping_number) == ping_number
        return record

    def _read_records(self, record_type: int) -> List[records.BaseRecord]:
        return self._reader.read_records_during_ping(
            record_type,
            self.sonar_settings.frame.time,
            self._next_ping_start,
            self._offset,
        )


class FileDataset:
    """Indexable dataset returning Pings from a 7k file.

    Provides random access into pings in a file with minimal overhead.

    Args:
        filename: Path to the s7k file, OR open file in binary mode, preferably with 0 buffering.
        include (PingType): Type of ping data we want to access

    """

    def __init__(
        self,
        file: Union[str, bytes, os.PathLike, BinaryIO],
        include: PingType = PingType.ANY,
        catalog_issue_handling: CatalogIssueHandling = CatalogIssueHandling.RAISE,
    ):
        """
        if include argument is not ANY, pings will be filtered.
        """
        self.pings = list(
            S7KFileReader(file, catalog_issue_handling).iter_pings(include)
        )
        self._ping_numbers = [p.ping_number for p in self.pings]

    @property
    def ping_numbers(self) -> List[int]:
        return self._ping_numbers

    def minimize_memory(self) -> None:
        for p in self.pings:
            p.minimize_memory()

    def __len__(self) -> int:
        return len(self.pings)

    def index_of(self, ping_number: int) -> int:
        return self._ping_numbers.index(ping_number)

    def get_by_number(
        self, ping_number: int, default: Optional[Ping] = None
    ) -> Optional[Ping]:
        if not isinstance(ping_number, int):
            raise TypeError("Ping number must be an integer")
        try:
            ping_index = self.ping_numbers.index(ping_number)
        except ValueError:
            return default
        return self.pings[ping_index]

    def __getitem__(self, index: Union[int, slice]) -> Union[Ping, List[Ping]]:
        return self.pings[index]


class ConcatDataset:
    """Concatenate a list of s7k dataset.

    The provided datasets are not assumed to be ordered and it will not order them.
    The ConcatDataset object provides seamless access to all pings in the datasets
    without the need for knowing each individual datasets size.

    Args:
        datasets (List): List of datasets to concatenate

    """

    def __init__(self, datasets):
        self.cum_lengths = np.cumsum([len(d) for d in datasets])
        self.datasets = datasets
        self._ping_numbers = [pn for ds in datasets for pn in ds.ping_numbers]

    def __len__(self) -> int:
        return self.cum_lengths[-1]

    @property
    def ping_numbers(self) -> List[int]:
        return self._ping_numbers

    def index_of(self, ping_number: int) -> int:
        return self.ping_numbers.index(ping_number)

    def get_by_number(
        self, ping_number: int, default: Optional[int] = None
    ) -> Optional[Ping]:
        if not isinstance(ping_number, int):
            raise TypeError("Ping number must be an integer")
        for ds in self.datasets:
            if (ping := ds.get_by_number(ping_number, default)) is not None:
                return ping
        return default

    def __iter__(self):
        for i in range(len(self)):
            self.__index = i
            yield self[i]
            self[i].minimize_memory()

    def minimize_memory(self):
        for ds in self.datasets:
            ds.minimize_memory()

    def __getitem__(self, index: Union[slice, int]) -> Union[Ping, List[Ping]]:
        if not isinstance(index, slice):
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
        else:
            return [self[i] for i in range(*index.indices(len(self)))]


class PingDataset(FileDataset):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "PingDataset has been renamed to FileDataset and will be removed in the next release",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)


class FolderDataset(ConcatDataset):
    """Read a folder of s7k files and create indexable s7k dataset.

    The FolderDataset is a subclass of the ConcatDataset. It assumes
    that the folderpath provided contains s7k files that are all part
    of the same voyage. It will therefore sort the files found in the
    folder and the indexing of the returned dataset will therefore be
    ordered by the first ping times.

    If provides the same ping access functionality as the FileDataset.

    NOTE: It is non-recursive, meaning it won't read from subdirectories.

    Args:
        folderpath (str): Path to directory of s7k files
        include (PingType): Types of pings we are interested in accessing

    """

    def __init__(
        self,
        folderpath: str,
        include: PingType = PingType.ANY,
        catalog_issue_handling=CatalogIssueHandling.RAISE,
    ):
        path = Path(folderpath)
        if isinstance(folderpath, str):
            # Check if it is a file, or directory
            if not path.is_dir():
                raise (
                    FileNotFoundError(
                        f"Provided folder '{folderpath}' could not be located"
                    )
                )
            filenames = list(path.glob("*.s7k"))

            if len(filenames) == 0:
                raise ValueError("Provided pathname did not match any files")
        else:
            raise TypeError("Pathname must be a string")

        datasets = []
        for f in filenames:
            datasets.append(
                FileDataset(
                    f, include=include, catalog_issue_handling=catalog_issue_handling
                )
            )

        # We should start by ordering the datasets by time
        self.datasets = sorted(datasets, key=lambda x: x[0].sonar_settings.frame.time)
        self._ping_numbers = []
        self.cum_lengths = []
        ds_count = 0
        # Loop over all the pings in the datasets and exclude duplicates.
        # This is necessary because of the way data is stored accross files,
        # meaning that subsequent files often include a couple of pings
        # from the end of the previous s7k file. This loop excludes all
        # duplicates.
        for ds in self.datasets:
            ds_pings = []
            for p in ds.pings:
                if p.ping_number not in self._ping_numbers:
                    ds_count += 1
                    ds_pings.append(p)
                    self._ping_numbers.append(p.ping_number)
                elif ds_pings and p.ping_number == ds_pings[0].ping_number:
                    # Old software sometimes wrote a duplicated 7000 record out of order
                    # at the beginning of a new file. We remove it here, to keep pings in order
                    del ds_pings[0]
                    ds_pings.append(p)

            ds.pings = ds_pings
            self.cum_lengths.append(ds_count)

        # Fix for issue #24
        for i in range(1, len(self.datasets)):
            # Get timestamp of first ping in subsequent dataset
            new_time = self.datasets[i][0].sonar_settings.frame.time

            # Update last ping of previous dataset
            self.datasets[i - 1][-1]._next_ping_start = new_time
