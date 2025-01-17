"""
Tools for reading structured binary data
"""
import abc
import io
import struct
from collections import defaultdict, namedtuple
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from . import records

_ElementTypes = namedtuple(
    "_ElementTypes",
    [
        # These are the canonical names of low-level element types:
        "c8",
        "i8",
        "u8",  # 3 int types of 1 byte
        "i16",
        "u16",  # 2 int types of 2 bytes
        "i32",
        "u32",  # 2 int types of 4 bytes
        "i64",
        "u64",  # 2 int types of 8 bytes
        "f32",  # 1 float type of 4 bytes
        "f64",  # 1 float type of 4 bytes
    ],
)
elemT = _ElementTypes(**dict(zip(_ElementTypes._fields, _ElementTypes._fields)))


def elemD_(name: str, fmt: str, count=1):
    return (name, (fmt, count))


def parse_7k_timestamp(bs: bytes) -> datetime:
    """Parse a timestamp from a bytes object"""
    # We have raw days, datetime takes days and months. Easier to just add them
    # as timedelta, and let datetime worry about leap-whatever
    y, d, s, h, m = struct.unpack("<HHfBB", bs)
    t = datetime(year=y, month=1, day=1)
    t += timedelta(
        # subtract 1 since datetime already starts at 1
        days=d - 1,
        hours=h,
        minutes=m,
        seconds=s,
    )
    return t


map_size_to_fmt = dict(
    (
        (elemT.c8, ("c", "B", 1)),
        (elemT.i8, ("b", "b", 1)),
        (elemT.u8, ("B", "u1", 1)),
        (elemT.i16, ("h", "i2", 2)),
        (elemT.u16, ("H", "u2", 2)),
        (elemT.i32, ("i", "i4", 4)),
        (elemT.u32, ("I", "u4", 4)),
        (elemT.i64, ("q", "i8", 8)),
        (elemT.u64, ("Q", "u8", 8)),
        (elemT.f32, ("f", "f4", 4)),
        (elemT.f64, ("d", "f8", 8)),
    )
)


class DataBlock(metaclass=abc.ABCMeta):
    """
    Reads fixed-size blocks of structured binary data, according to a specified format
    """

    _byte_order_fmt = "<"

    def __init__(self, elements):
        self._sizes = self._util_take_sizes(elements)
        self._names = self._util_take_names(elements)
        self._struct = self._util_create_struct(self._sizes)
        self._np_types = self._util_create_np_types(self._names, self._sizes)

    @property
    def size(self):
        return self._struct.size

    @property
    def numpy_types(self):
        return self._np_types

    def read(self, source: io.RawIOBase, count=1):
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Count is not int or non-positive?")

        dict_read = defaultdict(list)
        for _ in range(count):
            buf = source.read(self.size)
            if not buf:
                break
            unpacked = self._struct.unpack(buf)
            elements_zip = zip(self._names, self._sizes)
            offset = 0
            for name, (_, n_elems) in elements_zip:
                if isinstance(name, str):
                    dict_read[name] += unpacked[offset : (offset + n_elems)]
                offset += n_elems
        return {k: (v[0] if len(v) == 1 else tuple(v)) for k, v in dict_read.items()}

    def read_dense(self, source: io.RawIOBase, count=1) -> np.ndarray:
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Count is not int or non-positive?")
        dtype = np.dtype(self._np_types)
        if isinstance(source, io.FileIO):
            return np.fromfile(source, dtype=dtype, count=count)
        else:
            return np.frombuffer(
                source.read(dtype.itemsize * count), dtype=dtype, count=count
            )

    @staticmethod
    def _util_take_names(elements) -> tuple:
        return tuple(name for name, *_ in elements)

    @staticmethod
    def _util_take_sizes(elements) -> tuple:
        def f_take():
            for _, size in elements:
                if len(size) == 1:
                    size = size + (1,)
                yield size

        return tuple(f_take())

    @classmethod
    def _util_create_struct(cls, sizes) -> struct.Struct:
        fmts = [cls._byte_order_fmt]
        for type_name, count in sizes:
            count = "" if count == 1 else str(count)
            fmt, *_ = map_size_to_fmt[type_name]
            fmts.append(str(count) + fmt)
        return struct.Struct("".join(fmts))

    @classmethod
    def _util_create_np_types(cls, names, sizes):
        def f_name_fixer(idx, name):
            return f"__reserved{idx}__" if name is None else name

        bom = cls._byte_order_fmt
        types = []
        for idx, (name, (type_name, count)) in enumerate(zip(names, sizes)):
            name = f_name_fixer(idx, name)
            _, fmt, *_ = map_size_to_fmt[type_name]
            type_spec = [name, f"{bom}{fmt}"]
            if count > 1:
                type_spec += [(count,)]
            types.append(tuple(type_spec))
            del type_spec, fmt
            del idx, name, type_name, count
        return types  # apparently it should remain a list

    @staticmethod
    def _util_gen_elements(fields, names):
        results = []
        name_index = 0
        for field in fields:
            part_a, part_b, *_ = field
            if part_a is None:
                results.append(tuple([None, part_b]))
            else:
                results.append(tuple([names[name_index], part_a]))
                name_index += 1
        return results


# List of fields in a Data Record Frame.
# Using only primitive fields, for parsing.
DRF_PRIMITIVE_FIELDS = (
    "protocol_version",
    "offset",
    "sync_pattern",
    "size",
    "optional_data_offset",
    "optional_data_id",
    "time",
    "record_version",
    "record_type_id",
    "device_id",
    "system_enumerator",
    "flags",
)

# Removes partial time fields, and adds a proper time field.
DRF_REFINED_FIELDS = list(
    filter(lambda name: not name.startswith("time_"), DRF_PRIMITIVE_FIELDS)
) + ["time"]


class DRFBlock(DataBlock):
    """
    Reads a Data Record Frame from binary data.
    Specified in the Teledyne Reson Data Format Definition.
    """

    def __init__(self):
        elements = tuple(
            self._util_gen_elements(
                (
                    ((elemT.u16,), None),
                    ((elemT.u16,), None),
                    ((elemT.u32,), None),
                    ((elemT.u32,), None),
                    ((elemT.u32,), None),
                    ((elemT.u32,), None),
                    ((elemT.c8, 10), None),
                    ((elemT.u16,), None),
                    ((elemT.u32,), None),
                    ((elemT.u32,), None),
                    (None, (elemT.u16,)),
                    ((elemT.u16,), None),
                    (None, (elemT.u32,)),
                    ((elemT.u16,), None),
                    (None, (elemT.u16,)),
                    (None, (elemT.u32,)),
                    (None, (elemT.u32,)),
                    (None, (elemT.u32,)),
                ),
                names=DRF_PRIMITIVE_FIELDS,
            )
        )
        super().__init__(elements)

    def read(self, source: io.RawIOBase) -> Optional[records.DataRecordFrame]:
        init_data = super().read(source)
        # convert time from bytes to datetime
        if len(init_data):
            init_data["time"] = parse_7k_timestamp(b"".join(init_data["time"]))
            return records.DataRecordFrame(**init_data)
        else:
            return None
