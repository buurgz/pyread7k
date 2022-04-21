"""
Contains classes and functions for reading from 7k files.
Use FileDataset and FolderDataset to get high-level Ping objects.
Use S7kRecordReader or read_records for lower-level access.
"""
from ._datablock import DRFBlock
from ._datarecord import record
from ._ping import (
    CatalogIssueHandling,
    ConcatDataset,
    FileDataset,
    FolderDataset,
    Ping,
    PingDataset,
    PingType,
    S7KFileReader,
    S7KReader,
)
from ._recordreader import S7KRecordReader
from ._utils import *
