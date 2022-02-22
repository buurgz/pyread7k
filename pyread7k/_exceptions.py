"""
Custom exception handling for S7K files
"""


class CorruptFileCatalog(Exception):
    """Custom corrupt file catalog exception.
    This is used to raise exception when the file catalog is corrupt.
    Missing and corrupt file catalog should be handle differently,
    which is why we need both.

    """


class MissingFileCatalog(Exception):
    """Custom missing file catalog exception.
    This is used to raise exceptions when the filecatalog is missing.
    Missing and corrupt file catalog should be handle differently,
    which is why we need both.
    
    """


class CorruptFileHeader(Exception):
    """Custom corrupt file header exception.
    When there is an issue parsing the file header, this exception 
    is used.
    
    """


class UnsupportedRecordError(Exception):
    """Custom unsupported record exception.
    Raised when a record is not supported.
    
    """


class CorruptRecordDataError(Exception):
    """Raise an error when a records data is corrupted.
    This can be when a logging event was abruptly cancelled
    or if the record is corrupted in another way.
    
    """
