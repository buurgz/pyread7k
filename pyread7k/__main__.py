import logging
import os
import pathlib

import click
import requests
from dotenv import find_dotenv, load_dotenv, set_key

logger = logging.getLogger(__name__)

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
        logger.info(f"{output} already exists.")
        return None
    elif not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=False)

    logger.info(f"{output} did not exist. Downloading it from public S3 bucket.")

    # s3 = boto3.client("s3")
    # s3.download_file(_BUCKET, obj, str(p))
    with open(output, "wb") as fhandle:
        with requests.get(f"http://{_BUCKET}.s3.amazonaws.com/{obj}", stream=True) as r:
            fhandle.write(r.content)


@main.command()
def devsetup():
    """Instatiate the"""
    root = pathlib.Path(__file__).parent.parent

    if len(list(root.glob(".env"))) == 0:
        logger.info("No .env found. Creating one.")
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

    logger.info(
        "Data for development and testing is now available in the data directory."
    )


if __name__ == "__main__":
    main()
