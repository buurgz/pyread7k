""" Context specifier of the tests """
import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

bf_filepath = os.environ.get("bf_filepath", None)
ci_filepath = os.environ.get("ci_filepath", None)
if any([bf_filepath is None, ci_filepath is None]):
    raise FileNotFoundError(
        "Files for testing are not present. "
        "Follow the steps in the Getting Started section of the README to get the files."
    )
