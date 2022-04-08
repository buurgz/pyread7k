[![Teledyne Logo](https://raw.githubusercontent.com/Teledyne-Marine/pyread7k/master/images/TeledyneLogo.png)](teledynemarine.com)


# Pyread7k
Pyread7k is a library for reading 7k files. It provides a high-level interface to the data in a file, with an API that is a compromise between being ergonomic, while still being easy to correllate with the Data Format Definition.

Using the FileDataset or FolderDataset classes, files can be read on an as-needed basis with low overhead. Each Ping in the dataset contains the full set of records which are related to that specifc ping. This means you can read the amplitudes, settings, motion, etc. for any single ping without worrying about file offsets or any low-level details at all.

Low-level direct control is available with the _utils.py functions, which are reexported at top level. These allow you to read all records of a specific type manually

The 7k protocol is described in the [Data Format Definition](https://raw.githubusercontent.com/Teledyne-Marine/pyread7k/master/documents/DATA%20FORMAT%20DEFINITION%20-%207k%20Data%20Format.pdf) document.

# Installation
The library is published on PyPi, and can be installed using any python package manager, or if you want the latest you can install directly from the git repository.

```bash
# Pip
pip install pyread7k 
# Poetry
poetry add pyread7k
# Git
pip install git+https://github.com/Teledyne-Marine/pyread7k.git
```

## Building
If you want to build the package yourself, you can start by cloning the repository

```bash
git clone git+https://github.com/Teledyne-Marine/pyread7k.git
```

If you have poetry (and the right poetry version) you can build the wheel like so:

```bash
poetry build
```

Now the wheel is available in the dist directory

# Getting started
Working with the pyread7k library is quite intuitive, and given that you have a s7k file, you can load a dataset using the PingDataset class:
```python
import pyread7k

# Access files
dataset = pyread7k.FileDataset("path/to/file.s7k")
# Access to folders
dataset = pyread7k.FolderDataset("path/to_files/")
# Work with different files with a single interface
files = ["path1/to/file1.s7k", "path2/to/file2.s7k", "path3/to/file3.s7k"]
dataset = pyread7k.ConcatDataset([pyread7k.FileDataset(f) for f in files])
```

This gives you access to the pings, which consist of IQ or beamformed records, along with related data. All data is loaded on-demand when accessed:

```python
import numpy as np

for ping in dataset:
    if ping.has_beamformed:
        # Print mean amplitude for each ping with 7018 data
        print("Mean:", np.mean(ping.beamformed.amplitudes))
    # Print selected gain level for each ping
    print("Gain:", ping.sonar_settings.gain_selection)
```


# Dependencies

* `Python` 3.8 or later
* `psutil` 5.8.0
* `numpy` 1.20.1
* `geopy` 2.1.0


# Developing
It is easy to add new functionality, such as supporting more record types, to pyread7k. Get up and running:
- Install the [Poetry](https://python-poetry.org/docs/) dependency/package manager
- Clone the repositorty by executing `git clone git@github.com:Teledyne-Marine/pyread7k.git`
- Create a development environment by navigating to the repo folder and running `poetry install`
- Get some relevant data from the public data storage `poetry run python -m pyread7k devsetup`

Now you should be ready to add extra functionality.

# License
Copyright 2021 Teledyne RESON A/S
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
