"""
Custom exception handling for S7K files
"""


class CorruptFileCatalog(Exception):
    """Custom corrupt file catalog exception."""


class MissingFileCatalog(Exception):
    """Custom missing file catalog exception."""


class CorruptFileHeader(Exception):
    """Custom corrupt file catalog exception."""


class UnsupportedRecordError(Exception):
    """Custom unsupported record exception."""


class CorruptRecordDataError(Exception):
    """Raise an error when a records data is corrupted"""
