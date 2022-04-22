"""
Low-level classes for reading various 7k record types.
"""
# pylint: disable=invalid-name unnecessary-comprehension
import abc
import io
from typing import Any, Dict, Optional
from xml.etree import ElementTree as ET

import numpy as np

from . import records
from ._datablock import (
    DataBlock,
    DRFBlock,
    elemD_,
    elemT,
    map_size_to_fmt,
    parse_7k_timestamp,
)
from ._exceptions import (
    CorruptFileCatalog,
    CorruptFileHeader,
    CorruptRecordDataError,
    MissingFileCatalog,
)


def _bytes_to_str(dict, keys):
    """
    For each key, the corresponding dict value is transformed from
    a list of bytes to a string
    """
    for key in keys:
        byte_list = dict[key]
        termination = byte_list.index(b"\x00")
        dict[key] = b"".join(byte_list[:termination]).decode("UTF-8")


def _datablock_elemd(*items):
    """Maps the elemD function on arguments before passing to DataBlock"""
    return DataBlock(tuple(elemD_(*elems) for elems in items))


def _record_data_block(fields, data_field_size):
    """
    Some Record Data segments may have fields added in the future.
    To understand what fields are available, a field size must be read.
    If there are more fields in `fields` than data field size can handle, those
    are removed. If there are fewer, an unparsed field is added to `fields`.
    """
    field_size_accumulator = 0
    for index, field in enumerate(fields):
        if len(field) == 2:
            _name, elem_type = field
            count = 1
        else:
            _name, elem_type, count = field

        _, _, elem_size = map_size_to_fmt[elem_type]

        field_size_accumulator += count * elem_size

        if field_size_accumulator >= data_field_size:
            break
    if field_size_accumulator > data_field_size:
        # The forloop was broken, but sizes did not align
        raise CorruptRecordDataError(
            "Record data field lengths could not be matched to data field size"
        )
    available_fields = fields[: index + 1]
    if field_size_accumulator < data_field_size:
        # There are unknown fields added since this code was written.
        # These are added as a "reserved" field so that sizes line up.
        unknown_size = data_field_size - field_size_accumulator
        available_fields += ((None, elemT.c8, unknown_size),)

    return _datablock_elemd(*available_fields)


class DataRecord(metaclass=abc.ABCMeta):
    """
    Base class for all record readers.

    Subclasses provide functionality for reading specific records.
    These are NOT the classes returned to the library user, they are only readers.
    """

    _block_rth: DataBlock
    _block_drf = DRFBlock()
    _block_checksum = DataBlock((("checksum", ("u32",)),))
    implemented: Optional[Dict[int, Any]] = None
    _record_type_id = None

    def read(self, source: io.RawIOBase, drf: Optional[records.DataRecordFrame] = None):
        """Base record reader.

        Args:
            source (io.RawIOBase): The bytes object to read from
            drf (:obj: `records.DataRecordFrame`, optional): An optional input to minimize reads
        Returns:
            A data record

        """
        start_offset = source.tell()
        if drf is None:
            drf = self._block_drf.read(source)
        try:
            source.seek(start_offset)
            source.seek(4, io.SEEK_CUR)  # to sync pattern
            source.seek(drf.offset, io.SEEK_CUR)

            # If we are not parsing an implemented record
            # then we should just read whatever the drf says.
            # Otherwise we should check whether the size
            # from the drf and the size of the defined block
            # match. If they don't then we remove fields
            # from the elems until they do.
            parsed_data = self._read(source, drf, start_offset)

            checksum = self._block_checksum.read(source)["checksum"]
            if drf.flags & 0b1 > 0:  # Check if checksum is valid
                drf.checksum = checksum
            source.seek(start_offset)  # reset source to start

            return parsed_data
        except AttributeError as exc:
            if drf is None and self._record_type_id == 7300:
                raise CorruptFileCatalog(
                    "Corrupt file catalog (record 7300) or incorrect data block!"
                ) from exc
            if self._record_type_id == 7200:
                raise CorruptFileHeader(
                    "Corrupt file header (record 7200) or incorrect data block!"
                )
            raise exc  # If the error is unknown, propagate it
        except ValueError as exc:
            raise exc

    @classmethod
    def instance(cls, record_type_id: int):
        """Gets a specific datarecord by type id"""
        if cls.implemented is not None:
            return cls.implemented.get(record_type_id, _UnsupportedRecord())
        subclasses = cls.__subclasses__()
        cls.implemented = dict((c.record_type_id(), c()) for c in subclasses)
        return cls.implemented.get(record_type_id, _UnsupportedRecord())

    @abc.abstractmethod
    def _read(
        self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        # returns iterable of dicts:
        #    0: tuple of rth values (required)
        #    1: rd values (if not available, return None)
        #    2: od values (if not available, return None)
        raise NotImplementedError

    @classmethod
    def record_type_id(cls):
        """return data record type id"""
        return cls._record_type_id


class _DataRecord7000(DataRecord):

    """Sonar Settings"""

    _record_type_id = 7000

    _block_rth = DataBlock(
        (
            elemD_("sonar_id", elemT.u64),
            elemD_("ping_number", elemT.u32),
            elemD_("multi_ping_sequence", elemT.u16),
            elemD_("frequency", elemT.f32),
            elemD_("sample_rate", elemT.f32),
            elemD_("receiver_bandwidth", elemT.f32),
            elemD_("tx_pulse_width", elemT.f32),
            elemD_("tx_pulse_type_id", elemT.u32),
            elemD_("tx_pulse_envelope_id", elemT.u32),
            elemD_("tx_pulse_envelope_parameter", elemT.f32),
            elemD_("tx_pulse_mode", elemT.u16),
            elemD_(None, elemT.u16),
            elemD_("max_ping_rate", elemT.f32),
            elemD_("ping_period", elemT.f32),
            elemD_("range_selection", elemT.f32),
            elemD_("power_selection", elemT.f32),
            elemD_("gain_selection", elemT.f32),
            elemD_("control_flags", elemT.u32),
            elemD_("projector_id", elemT.u32),
            elemD_("projector_beam_angle_vertical", elemT.f32),
            elemD_("projector_beam_angle_horizontal", elemT.f32),
            elemD_("projector_beam_width_vertical", elemT.f32),
            elemD_("projector_beam_width_horizontal", elemT.f32),
            elemD_("projector_beam_focal_point", elemT.f32),
            elemD_("projector_beam_weighting_window_type", elemT.u32),
            elemD_("projector_beam_weighting_window_parameter", elemT.f32),
            elemD_("transmit_flags", elemT.u32),
            elemD_("hydrophone_id", elemT.u32),
            elemD_("receive_beam_weighting_window", elemT.u32),
            elemD_("receive_beam_weighting_parameter", elemT.f32),
            elemD_("receive_flags", elemT.u32),
            elemD_("receive_beam_width", elemT.f32),
            elemD_("bottom_detection_filter_min_range", elemT.f32),
            elemD_("bottom_detection_filter_max_range", elemT.f32),
            elemD_("bottom_detection_filter_min_depth", elemT.f32),
            elemD_("bottom_detection_filter_max_depth", elemT.f32),
            elemD_("absorption", elemT.f32),
            elemD_("sound_velocity", elemT.f32),
            elemD_("spreading", elemT.f32),
            elemD_(None, elemT.u16),
        )
    )

    def _read(
        self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = self._block_rth.read(source)
        return records.SonarSettings(**rth, frame=drf)


class _DataRecord7001(DataRecord):
    """Configuration"""

    _record_type_id = 7001

    _block_rth = DataBlock(
        (
            elemD_("sonar_serial_number", elemT.u64),
            elemD_("number_of_devices", elemT.u32),
        )
    )

    _block_rd_info = DataBlock(
        (
            elemD_("identifier", elemT.u32),
            elemD_("description", elemT.c8, 60),  # We should parse this better
            elemD_("alphadata_card", elemT.u32),
            elemD_("serial_number", elemT.u64),
            elemD_("info_length", elemT.u32),
        )
    )

    def _read(
        self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = self._block_rth.read(source)
        rd = []
        for _ in range(rth["number_of_devices"]):
            device_data = self._block_rd_info.read(source)
            _bytes_to_str(device_data, ["description"])
            xml_string = source.read(device_data["info_length"])
            # Indexing removes a weird null-termination
            device_data["info"] = ET.fromstring(xml_string[:-1])
            rd.append(records.DeviceConfiguration(**device_data))

        return records.Configuration(**rth, devices=rd, frame=drf)


class _DataRecord7200(DataRecord):

    _record_type_id = 7200

    _block_rth = DataBlock(
        (
            elemD_("file_id", elemT.u64, 2),
            elemD_("version_number", elemT.u16),
            elemD_(None, elemT.u16),
            elemD_("session_id", elemT.u64, 2),
            elemD_("record_data_size", elemT.u32),
            elemD_("number_of_devices", elemT.u32),
            elemD_("recording_name", elemT.c8, 64),
            elemD_("recording_program_version_number", elemT.c8, 16),
            elemD_("user_defined_name", elemT.c8, 64),
            elemD_("notes", elemT.c8, 128),
        )
    )

    _block_rd_device_type = DataBlock(
        (elemD_("device_ids", elemT.u32), elemD_("system_enumerators", elemT.u16))
    )

    _block_od = DataBlock(
        (elemD_("catalog_size", elemT.u32), elemD_("catalog_offset", elemT.u64))
    )

    def _read(
        self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = self._block_rth.read(source)
        _bytes_to_str(
            rth,
            [
                "recording_name",
                "recording_program_version_number",
                "user_defined_name",
                "notes",
            ],
        )
        try:
            rd = self._block_rd_device_type.read(source, rth["number_of_devices"])
            source.seek(start_offset)
            source.seek(drf.optional_data_offset, io.SEEK_CUR)
            od = self._block_od.read(source)
        except Exception as exc:
            raise MissingFileCatalog(
                "The optional data of the file header (record 7200) is not present."
            ) from exc

        # return rth, rd, od
        return records.FileHeader(**rth, **rd, **od, frame=drf)


class _DataRecord7300(DataRecord):

    _record_type_id = 7300

    _block_rth = DataBlock(
        (
            elemD_("size", elemT.u32),
            elemD_("version", elemT.u16),
            elemD_("number_of_records", elemT.u32),
            elemD_(None, elemT.u32),
        )
    )

    _block_rd_entry = DataBlock(
        (
            elemD_("sizes", elemT.u32),
            elemD_("offsets", elemT.u64),
            elemD_("record_types", elemT.u16),
            elemD_("device_ids", elemT.u16),
            elemD_("system_enumerators", elemT.u16),
            elemD_("times", elemT.c8, 10),
            elemD_("record_counts", elemT.u32),
            elemD_(None, elemT.u16, 8),
        )
    )

    def _read(
        self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = self._block_rth.read(source)
        rd = self._block_rd_entry.read(source, rth["number_of_records"])
        times_bytes = rd["times"]
        rd["times"] = tuple(
            parse_7k_timestamp(b"".join(times_bytes[i : i + 10]))
            for i in range(0, len(times_bytes), 10)
        )
        return records.FileCatalog(**rth, **rd, frame=drf)


class _DataRecord7004(DataRecord):
    """Beam Geometry"""

    _record_type_id = 7004
    _block_rth = DataBlock(
        (elemD_("sonar_id", elemT.u64), elemD_("number_of_beams", elemT.u32))
    )

    def _read(
        self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = self._block_rth.read(source)
        n_beams = rth["number_of_beams"]
        block_rd_size = (
            drf.size
            - self._block_drf.size
            - self._block_checksum.size
            - self._block_rth.size
        )
        block_rd_elements = (
            elemD_("vertical_angles", elemT.f32, n_beams),
            elemD_("horizontal_angles", elemT.f32, n_beams),
            elemD_("beam_width_ys", elemT.f32, n_beams),
            elemD_("beam_width_xs", elemT.f32, n_beams),
            elemD_("tx_delays", elemT.f32, n_beams),
        )
        block_rd = DataBlock(block_rd_elements)
        if block_rd.size != block_rd_size:
            # tx_delays missing
            block_rd = DataBlock(block_rd_elements[:-1])
            assert block_rd.size == block_rd_size, (block_rd.size, block_rd_size)

        array_rd = block_rd.read_dense(source)
        # Convert to dictionary
        rd = {k[0]: array_rd[k[0]].squeeze() for k in block_rd.numpy_types}
        return records.BeamGeometry(**rth, **rd, frame=drf)


class _DataRecord7010(DataRecord):
    """TVG Values"""

    _record_type_id = 7010
    _block_rth = DataBlock(
        (
            elemD_("sonar_id", elemT.u64),
            elemD_("ping_number", elemT.u32),
            elemD_("multi_ping_sequence", elemT.u16),
            elemD_("number_of_samples", elemT.u32),
            elemD_(None, elemT.u32, 8),
        )
    )

    _block_gain_sample = DataBlock((elemD_("gains", elemT.f32),))

    def _read(
        self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = self._block_rth.read(source)
        sample_count = rth["number_of_samples"]
        rd = self._block_gain_sample.read_dense(source, sample_count)
        return records.TVG(**rth, gains=rd["gains"], frame=drf)


class _DataRecord7018(DataRecord):
    """Beamformed data"""

    _record_type_id = 7018
    _block_rth = DataBlock(
        (
            elemD_("sonar_id", elemT.u64),
            elemD_("ping_number", elemT.u32),
            elemD_("is_multi_ping", elemT.u16),
            elemD_("number_of_beams", elemT.u16),
            elemD_("number_of_samples", elemT.u32),
            elemD_(None, elemT.u32, 8),
        )
    )

    _block_rd_amp_phs = DataBlock((elemD_("amp", elemT.u16), elemD_("phs", elemT.i16)))

    def _read(
        self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        _start_offset = source.tell()
        rth = self._block_rth.read(source)
        n_samples = rth["number_of_samples"]
        n_beams = rth["number_of_beams"]
        count = n_samples * n_beams
        try:
            rd = self._block_rd_amp_phs.read_dense(source, count)
            rd = rd.reshape((n_samples, n_beams))

            return records.Beamformed(
                **rth, amplitudes=rd["amp"], phases=rd["phs"], frame=drf
            )
        except ValueError as exc:
            raise CorruptRecordDataError(
                f"Record {self._record_type_id} at {_start_offset} has corrupt record data!"
            ) from exc


class _DataRecord7027(DataRecord):
    """Raw Detection data"""

    _record_type_id = 7027
    _block_rth = _datablock_elemd(
        ("sonar_id", elemT.u64),
        ("ping_number", elemT.u32),
        ("multi_ping_sequence", elemT.u16),
        ("detection_count", elemT.u32),
        ("data_field_size", elemT.u32),
        ("detection_algorithm", elemT.u8),
        ("_flags", elemT.u32),
        ("sampling_rate", elemT.f32),
        ("tx_angle", elemT.f32),
        ("applied_roll", elemT.f32),
        (None, elemT.u32, 15),
    )

    _rd_fields = (
        ("beam_descriptor", elemT.u16),
        ("detection_point", elemT.f32),
        ("rx_angle", elemT.f32),
        ("flags", elemT.u32),
        ("quality", elemT.u32),
        ("uncertainty", elemT.f32),
        ("intensity", elemT.f32),
        ("min_limit", elemT.f32),
        ("max_limit", elemT.f32),
    )

    def _read(
        self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = self._block_rth.read(source)
        rth["detection_algorithm"] = records.DetectionAlgorithm(
            rth["detection_algorithm"]
        )

        # The block_rd may change based on data_field_size
        block_rd = _record_data_block(self._rd_fields, rth["data_field_size"])
        rd = block_rd.read_dense(source, rth["detection_count"])

        return records.RawDetectionData(**rth, detections=rd, frame=drf)


class _DataRecord7028(DataRecord):
    """Snippet data"""

    _record_type_id = 7028
    _block_rth = DataBlock(
        (
            elemD_("sonar_id", elemT.u64),
            elemD_("ping_number", elemT.u32),
            elemD_("multi_ping_sequence", elemT.u16),
            elemD_("detection_count", elemT.u16),
            elemD_("error_flag", elemT.u8),
            elemD_("control_flags", elemT.u8),
            elemD_("flags", elemT.u32),
            elemD_(None, elemT.u32, 6),
        )
    )
    _block_rd = DataBlock(
        (
            elemD_("beam_number", elemT.u16),
            elemD_("snippet_start", elemT.u32),
            elemD_("detection_sample", elemT.u32),
            elemD_("snippet_end", elemT.u32),
        )
    )

    def _read(
        self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = self._block_rth.read(source)
        rth["control_flags"] = records.SnippetControlFlag(rth["control_flags"])

        rd = self._block_rd.read_dense(source, rth["detection_count"])

        # Pre-read all intensities into linear array for speed
        snippet_type = elemT.u32 if rth["flags"] & 1 == 1 else elemT.u16
        snippet_block = DataBlock((elemD_("intensity", snippet_type),))
        snippet_lengths = rd["snippet_end"] - rd["snippet_start"]
        all_intensities = snippet_block.read_dense(source, int(snippet_lengths.sum()))

        # Partition intensities into snippets
        offset = 0
        intensities = []
        for snippet_length in snippet_lengths:
            intensities.append(all_intensities[offset : offset + snippet_length])
            offset += snippet_length

        return records.SnippetData(
            **rth, bottom_detections=rd, intensities=intensities, frame=drf
        )


class _DataRecord7038(DataRecord):
    """IQ data"""

    _record_type_id = 7038
    _block_rth = DataBlock(
        (
            elemD_("serial_number", elemT.u64),  # Sonar serial number
            elemD_("ping_number", elemT.u32),  # Sequential number
            elemD_(None, elemT.u16),  # Reserved (zeroed) but see note 1 below
            elemD_("channel_count", elemT.u16),  # Num system Rx elements
            elemD_("n_samples", elemT.u32),  # Num samples within ping
            elemD_("n_actual_channels", elemT.u16),  # Num elems in record
            elemD_("start_sample", elemT.u32),  # First sample in record
            elemD_("stop_sample", elemT.u32),  # Last sample in record
            elemD_("sample_type", elemT.u16),  # Sample type ID
            elemD_(None, elemT.u32, 7),
        )
    )  # Reserved (zeroed)

    # Note 1: Original DFD20724.docx document defines this element as
    # 'Reserved u16'. The MATLAB reader parses this as "multipingSequence".
    # This implementation follows the document and sets as reserved.

    _block_rd_data_u16 = DataBlock((elemD_("amp", elemT.u16), elemD_("phs", elemT.i16)))

    def _read(
        self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = self._block_rth.read(source)

        n_actual_channels = rth["n_actual_channels"]

        block_channel_array = DataBlock(
            (elemD_("channel_array", elemT.u16, n_actual_channels),)
        )

        channel_array = block_channel_array.read_dense(source)
        channel_array = np.squeeze(channel_array["channel_array"])
        rth["channel_array"] = channel_array

        n_actual_samples = rth["stop_sample"] - rth["start_sample"] + 1
        sample_type = rth["sample_type"]

        def f_block_actual_data(elem_type):
            return DataBlock(
                (
                    elemD_(
                        "actual_data",
                        elem_type,
                        n_actual_channels * n_actual_samples * 2,
                    ),
                )
            )

        # From document DFD20724.docx:
        # System data is always 16 bits I & Q. Sample type is used only for
        # the purpose of the compatibility with older tools. The following
        # values can be contained by the field:
        #    12 – Data is reported as i16 I and i16 Q aligned with four least
        #         significant bits truncated and aligned by LSB.
        #    16 – Data is reported as i16 I and i16 Q as received from Rx HW.
        #     8 – Data is reported as i8 I and i8 Q. Only most significant
        #         eight bits of 16-bit data are used.
        #     0 – Indicates that the data is not valid.

        if sample_type == 8:
            # from MATLAB reader:
            actual_data = f_block_actual_data(elemT.i8).read_dense(source)
            actual_data = np.squeeze(actual_data["actual_data"])
            actual_data[actual_data < 0] += 65536
            actual_data *= 16
            actual_data[actual_data > 2047] -= 4096
        elif sample_type == 16:
            actual_data = f_block_actual_data(elemT.i16).read_dense(source)
            actual_data = np.squeeze(actual_data["actual_data"])
        else:
            # Data is either invalid (0) or 12 bit (not supported):
            rd = dict(value=f"Unsupported sample type ID {sample_type}")
            return rth, rd, None  # <-- early RETURN

        rd_value = np.zeros(
            (rth["n_samples"], n_actual_channels),
            dtype=[(elem, actual_data.dtype.name) for elem in ("i", "q")],
        )

        rd_view = rd_value[rth["start_sample"] : rth["stop_sample"] + 1, :]
        rd_view["i"][:, channel_array] = actual_data[0::2].reshape(
            (-1, n_actual_channels)
        )
        rd_view["q"][:, channel_array] = actual_data[1::2].reshape(
            (-1, n_actual_channels)
        )

        return records.RawIQ(**rth, iq=rd_value, frame=drf)


class _DataRecord1003(DataRecord):
    """Position - GPS Coordinates"""

    _record_type_id = 1003
    _block_rth = DataBlock(
        (
            elemD_("datum_id", elemT.u32),
            elemD_("latency", elemT.f32),
            elemD_("latitude_northing", elemT.f64),
            elemD_("longitude_easting", elemT.f64),
            elemD_("height", elemT.f64),
            elemD_("position_type", elemT.u8),
            elemD_("utm_zone", elemT.u8),
            elemD_("quality_flag", elemT.u8),
            elemD_("positioning_method", elemT.u8),
            elemD_("number_of_satellites", elemT.u8),
        )
    )

    def _read(
        self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = self._block_rth.read(source)
        return records.Position(**rth, frame=drf)


class _DataRecord1012(DataRecord):
    """Roll Pitch Heave"""

    _record_type_id = 1012
    _block_rth = DataBlock(
        (
            elemD_("roll", elemT.f32),
            elemD_("pitch", elemT.f32),
            elemD_("heave", elemT.f32),
        )
    )

    def _read(
        self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = self._block_rth.read(source)
        return records.RollPitchHeave(**rth, frame=drf)


class _DataRecord1008(DataRecord):
    """Depth"""

    _record_type_id = 1008
    _block_rth = DataBlock(
        (
            elemD_("depth_descriptor", elemT.u8),
            elemD_("correction_flag", elemT.u8),
            elemD_(None, elemT.u16),  # Reserved
            elemD_("depth", elemT.f32),
        )
    )

    def _read(
        self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = self._block_rth.read(source)
        return records.Depth(**rth, frame=drf)


class _DataRecord1013(DataRecord):
    """Heading"""

    _record_type_id = 1013
    _block_rth = DataBlock((elemD_("heading", elemT.f32),))

    def _read(
        self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = self._block_rth.read(source)
        rd = None  # no rd
        od = None  # no optional data
        return records.Heading(**rth, frame=drf)


class _DataRecord1017(DataRecord):
    """PanTiltRoll"""

    _record_type_id = 1017
    _block_rth = _datablock_elemd(
        ("pan", elemT.f32),
        ("tilt", elemT.f32),
        ("roll", elemT.f32),
        ("pan_error", elemT.u32),
        ("tilt_error", elemT.u32),
        ("roll_error", elemT.u32),
    )

    def _read(
        self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = self._block_rth.read(source)
        return records.PanTiltRoll(**rth, frame=drf)


class _DataRecord1018(DataRecord):
    """Velocity"""

    _record_type_id = 1018
    _block_rth = DataBlock(
        (
            elemD_("velocity_x", elemT.f32),
            elemD_("velocity_y", elemT.f32),
            elemD_("velocity_z", elemT.f32),
        )
    )

    def _read(
        self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = self._block_rth.read(source)
        return records.Velocity(**rth, frame=drf)


class _DataRecord7022(DataRecord):
    """Sonar Source Version"""

    _record_type_id = 7022
    _block_rth = DataBlock((elemD_("sonar_source_version", elemT.c8, 32),))

    def _read(
        self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth = self._block_rth.read(source)
        _bytes_to_str(rth, ["sonar_source_version"])
        return records.SonarSourceVersion(**rth, frame=drf)


class _UnsupportedRecord(DataRecord):
    """Unsupported"""

    def _read(
        self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        record_bytes = source.read(
            drf.size - self._block_drf.size - self._block_checksum.size
        )

        return records.UnsupportedRecord(
            frame=drf, record_type=drf.record_type_id, record_bytes=record_bytes
        )


class _DataRecord7503(DataRecord):

    """Remote Control Sonar Settings"""

    _record_type_id = 7503

    _fields = (
        ("sonar_id", elemT.u64),
        ("ping_number", elemT.u32),
        ("frequency", elemT.f32),
        ("sample_rate", elemT.f32),
        ("receiver_bandwidth", elemT.f32),
        ("tx_pulse_width", elemT.f32),
        ("tx_pulse_type_id", elemT.u32),
        ("tx_pulse_envelope_id", elemT.u32),
        ("tx_pulse_envelope_parameter", elemT.f32),
        ("tx_pulse_mode", elemT.u16),
        (None, elemT.u16),
        ("max_ping_rate", elemT.f32),
        ("ping_period", elemT.f32),
        ("range_selection", elemT.f32),
        ("power_selection", elemT.f32),
        ("gain_selection", elemT.f32),
        ("control_flags", elemT.u32),
        ("projector_id", elemT.u32),
        ("projector_beam_angle_vertical", elemT.f32),
        ("projector_beam_angle_horizontal", elemT.f32),
        ("projector_beam_width_vertical", elemT.f32),
        ("projector_beam_width_horizontal", elemT.f32),
        ("projector_beam_focal_point", elemT.f32),
        ("projector_beam_weighting_window_type", elemT.u32),
        ("projector_beam_weighting_window_parameter", elemT.f32),
        ("transmit_flags", elemT.u32),
        ("hydrophone_id", elemT.u32),
        ("receive_beam_weighting_window", elemT.u32),
        ("receive_beam_weighting_parameter", elemT.f32),
        ("receive_flags", elemT.u32),
        ("bottom_detection_filter_min_range", elemT.f32),
        ("bottom_detection_filter_max_range", elemT.f32),
        ("bottom_detection_filter_min_depth", elemT.f32),
        ("bottom_detection_filter_max_depth", elemT.f32),
        ("absorption", elemT.f32),
        ("sound_velocity", elemT.f32),
        ("spreading", elemT.f32),
        ("vernier_operation_mode", elemT.u8),
        ("automatic_filter_window", elemT.u8),
        ("tx_array_position_offset_x", elemT.f32),
        ("tx_array_position_offset_y", elemT.f32),
        ("tx_array_position_offset_z", elemT.f32),
        ("head_tilt_x", elemT.f32),
        ("head_tilt_y", elemT.f32),
        ("head_tilt_z", elemT.f32),
        ("ping_state", elemT.u32),
        ("beam_spacing_mode", elemT.u16),
        ("sonar_source_mode", elemT.u16),
        ("adaptive_gate_bottom_filter_min", elemT.f32),
        ("adaptive_gate_bottom_filter_max", elemT.f32),
        ("trigger_out_width", elemT.f64),
        ("trigger_out_offset", elemT.f64),
        ("xx_series_projector_selection", elemT.u16),
        (None, elemT.u32, 2),
        ("xx_series_alternate_gain", elemT.f32),
        ("vernier_filter", elemT.u8),
        (None, elemT.u8),
        ("custom_beams", elemT.u16),
        ("coverage_angle", elemT.f32),
        ("coverage_mode", elemT.u8),
        ("quality_filter_flags", elemT.u8),
        ("horizontal_receiver_beam_steering_angle", elemT.f32),
        ("flexmode_sector_coverage", elemT.f32),
        ("flexmode_sector_steering", elemT.f32),
        ("constant_spacing", elemT.f32),
        ("beam_mode_selection", elemT.u16),
        ("depth_gate_tilt", elemT.f32),
        ("applied_frequency", elemT.f32),
        ("element_number", elemT.u32),
        ("max_image_height", elemT.u32),
        ("bytes_per_pixel", elemT.u32),
    )
    _block_rth = _datablock_elemd(*_fields)

    def _read(
        self, source: io.RawIOBase, drf: records.DataRecordFrame, start_offset: int
    ):
        rth_size = self._block_rth.size
        source_rth_size = drf.size - self._block_drf.size - self._block_checksum.size
        if rth_size != source_rth_size:
            block_rth = _record_data_block(self._fields, source_rth_size)
        else:
            block_rth = self._block_rth

        rth = block_rth.read(source)
        return records.RemoteControlSonarSettings(**rth, frame=drf)


def record(type_id: int) -> DataRecord:
    """Get a s7k record reader by record id"""
    return DataRecord.instance(type_id)
