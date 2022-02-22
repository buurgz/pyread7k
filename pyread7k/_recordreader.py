from typing import List, Generator
from pathlib import Path
from . import DRFBlock, _datarecord, records
import logging

logger = logging.getLogger(__name__)


def S7KRecordReader(filename: str, records_to_read: List[int] = []) -> Generator:
    """Linearly parse s7k files.
    The S7KRecordReader is a generator which linearly goes through the s7k 
    file provided in the input argument and returns one record at a time.
    For records that haven't been implemented yet, it will return
    an UnsupportedRecord, which will just include the data record frame.

    Args:
        filename (str): Name of the file to read
        records_to_read (List[int]): List of records to parse. 
            Default is the empty list representing all.

    Returns:
        A datarecord or unsupported record
    
    """

    # Ensure that the provided filepath is a string
    if not isinstance(filename, str):
        raise TypeError("Filename is not a string")

    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(f"Filename '{filename}' could not be found!")
    with path.open(mode="rb", buffering=0) as fhandle:
        offset = 0
        while True:
            drf = DRFBlock().read(fhandle)
            if drf is None:
                break

            fhandle.seek(offset)
            if (not (drf.record_type_id in records_to_read)) and len(records_to_read):
                offset += drf.size
                fhandle.seek(offset)
            else:
                try:
                    record = _datarecord.record(drf.record_type_id).read(fhandle, drf)
                except _datarecord.UnsupportedRecordError as exc:
                    logger.info(exc)
                    record = records.UnsupportedRecord(drf, drf.record_type_id)
                except Exception as exc:
                    # TODO: Do we want to be able to read corrupt files? If so we can just warn about the corrupt file and continue
                    fhandle.close()
                    raise exc
                offset += drf.size
                fhandle.seek(offset)
                yield record
