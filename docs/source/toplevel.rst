========
Main module
========

.. To use autodoc on reexports from __init__ we must either declare all members explicitly, or declare
.. an __all__ on the __init__ module. We choose explicit declarations for now, maybe it should be changed
.. when the module is better organized.

.. We use :noindex: to prevent the reexports from having to reference points in the docs.
.. automodule:: pyread7k
    :noindex: 
    :members: record, CatalogIssueHandling, ConcatDataset, FileDataset, FolderDataset, Ping,
        PingType, S7KFileReader, S7KReader, S7KRecordReader, read_file_header, read_file_catalog, get_record_offsets,
        get_record_count, gen_records, read_records, export_catalog,