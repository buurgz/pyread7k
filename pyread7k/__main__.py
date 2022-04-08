import os
import pathlib

import boto3
import click
from dotenv import find_dotenv, load_dotenv, set_key

_BUCKET = "tdy-marine.example-7k"
_DATADIR = pathlib.Path(".") / "data"
_REQUIRED_ENVIRONMENT_FILES = {
    "bf_filepath": "T50-P/with beamformed data/NBS-Snippets-Sensor-WC.s7k",
    "ci_filepath": "T50-P/without beamformed data/Bathy-Snippets-NormalizedBS.s7k",
}


@click.group()
def main():
    """Entrypoint for pyread7k"""


def getdata(obj, output, overwrite=False):
    p = pathlib.Path(output)
    if p.exists() and not overwrite:
        return None
    elif not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=False)

    s3 = boto3.client("s3")
    s3.download_file(_BUCKET, obj, str(p))


@main.command()
def devsetup():
    """Instatiate the """
    root = pathlib.Path(__file__).parent.parent
    if len(list(root.glob(".env"))) == 0:
        # Create the .env
        with (root / ".env").open("w") as _:
            pass

        # Check if the data directory exists
        for (envkey, filename) in _REQUIRED_ENVIRONMENT_FILES.items():
            filepath = _DATADIR / filename
            getdata(filename, filepath)
            set_key(str(root / ".env"), envkey, str(filepath.absolute()))

    # Load the environment variables
    load_dotenv(find_dotenv())

    for (envkey, filename) in _REQUIRED_ENVIRONMENT_FILES.items():
        filepath = os.environ.get(envkey, "")
        # If the filepath is empty then set the key
        # if the filepath doesn't exist download the one in the default
        if (filepath == "") or (not os.path.exists(filepath)):
            filepath = _DATADIR / filename
            getdata(filename, filepath)
            set_key(str(root / ".env"), envkey, str(filepath.absolute()))


if __name__ == "__main__":
    main()
