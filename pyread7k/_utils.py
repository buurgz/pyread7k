import collections
import csv
import functools
import io
import itertools as it
from copy import copy
from typing import Iterable, Iterator, Tuple, TypeVar

from . import records
from ._datablock import DRFBlock
from ._datarecord import record as _record
from .records import DataRecordFrame, FileCatalog, FileHeader

__all__ = [
    "read_file_header",
    "read_file_catalog",
    "get_record_offsets",
    "get_record_count",
    "gen_records",
    "read_records",
    "export_catalog",
]


T = TypeVar("T")


def window(seq: Iterable[T], n: int) -> Iterator[Tuple[T, ...]]:
    """Return a sliding window of width n over data from the iterable
    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    """
    iterator = iter(seq)
    q = collections.deque(it.islice(iterator, n), maxlen=n)
    if len(q) == n:
        yield tuple(q)
    for elem in iterator:
        q.append(elem)
        yield tuple(q)


def cached_property(func):
    """
    Fix functools.cached_property to preserve docstrings and name.
    Note that it does not properly preserve type hints!
    """
    return functools.update_wrapper(functools.cached_property(func), func)


def read_file_header(source: io.RawIOBase) -> FileHeader:
    """Read the file header 7200 record"""
    return _record(7200).read(source)


def read_file_catalog(source: io.RawIOBase, file_header: FileHeader) -> FileCatalog:
    """Read the file catalog 7300 record"""
    source.seek(file_header.catalog_offset)
    file_catalog: FileCatalog = _record(7300).read(source)
    return file_catalog


def build_file_catalog(source: io.RawIOBase) -> FileCatalog:
    """Build the file catalog using linear reading of s7k file.

    This utility function is used to construct a file catalog by reading through
    the records in the s7k file. It currently correctly handles all but:
    record_counts.

    """
    file_catalog_data = {
        "sizes": [],
        "offsets": [],
        "record_types": [],
        "device_ids": [],
        "system_enumerators": [],
        "times": [],
        "record_counts": [],
    }
    source.seek(0)
    number_of_records = 0
    offset = 0
    frame = None

    drf_dummy = DRFBlock()
    DRF_BYTE_SIZE = drf_dummy.size

    drf = DRFBlock().read(source)

    while True:
        if drf.record_type_id != 7300:
            file_catalog_data["offsets"].append(offset)
            file_catalog_data["sizes"].append(drf.size)
            file_catalog_data["record_types"].append(drf.record_type_id)
            file_catalog_data["device_ids"].append(drf.device_id)
            file_catalog_data["system_enumerators"].append(drf.system_enumerator)
            file_catalog_data["times"].append(drf.time)
            # TODO: Fix as this does not work as intended right now
            fragmented = int(drf.size > 60000)
            file_catalog_data["record_counts"].append(fragmented)
            number_of_records += 1
            frame = drf

        offset += drf.size

        # Read the full record plus the next drf
        raw_bytes = source.read(drf.size)
        if len(raw_bytes) < drf.size:
            break

        drf = DRFBlock().read(io.BytesIO(raw_bytes[-DRF_BYTE_SIZE:]))

    source.seek(0)
    file_catalog_data["number_of_records"] = number_of_records
    # Create a dummy frame for the file catalog
    dummy_frame = frame
    # Calculate the size of the record frames data
    dummy_frame.size = (8 + 4 + 2 + 2 + 2 + 10 + 4 + 8 * 2) * number_of_records
    # Add the size of the drf
    dummy_frame.size += DRF_BYTE_SIZE
    # Add the size of the record header of 7300
    dummy_frame.size += 4 + 4 + 4 + 2
    # Add the size of the checksum
    dummy_frame.size += 4

    dummy_frame.version = 1
    dummy_frame.record_type_id = 7300
    dummy_frame.system_enumerator = 0
    dummy_frame.flags = 0
    dummy_frame.checksum = None
    dummy_frame.device_id = 7000  # System event id
    file_catalog_data["frame"] = dummy_frame
    file_catalog_data["size"] = 14
    file_catalog_data["version"] = 1
    return FileCatalog(**file_catalog_data)


def get_record_offsets(type_id: int, file_catalog: FileCatalog) -> tuple:
    """Get offsets to all records of given type_id from the catalog"""

    cat_zip = zip(file_catalog.offsets, file_catalog.record_types)

    return tuple(offset for offset, _type_id in cat_zip if _type_id == type_id)


def get_record_count(type_id: int, file_catalog: FileCatalog) -> int:
    """Count number of records of given type in the catalog"""
    return len(get_record_offsets(type_id, file_catalog))


def gen_records(
    type_id: int,
    source: io.RawIOBase,
    file_catalog: FileCatalog,
    *,
    first_idx=0,
    count=None,
):
    """Generator reading records of the given type from the file"""
    start_offset = source.tell()
    cat_offsets = get_record_offsets(type_id, file_catalog)
    if first_idx > 0:
        cat_offsets = cat_offsets[first_idx:]

    for idx, offset in enumerate(cat_offsets):
        if count is not None and idx >= count:
            break
        source.seek(offset)
        data = _record(type_id).read(source)
        source.seek(start_offset)  # reset source
        yield data


def read_records(
    type_id: int,
    source: io.RawIOBase,
    file_catalog: FileCatalog,
    *,
    first_idx=0,
    count=None,
) -> records.BaseRecord:
    """Read all records of the given type from the file"""

    return tuple(
        gen_records(type_id, source, file_catalog, first_idx=first_idx, count=count)
    )


def export_catalog(filename: str, file_catalog: FileCatalog):
    """Write the catalog to a file in csv format"""

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=";", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow([f"file={filename}"])
        writer.writerow(["idx", "record_id", "file_offset", "size"])
        for idx, (type_id, offset, size) in enumerate(
            zip(
                file_catalog.record_types,
                file_catalog.offsets,
                file_catalog.sizes,
            )
        ):
            writer.writerow(str(n) for n in [idx, type_id, offset, size])
