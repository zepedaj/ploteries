from .main import main, path_arg
from ploteries.data_store import DataStore
from rich import print

from tqdm import tqdm
import climax as clx
from time import sleep
from itertools import islice
import os.path as osp
import numpy as np
from ploteries.writer import Writer
from contextlib import nullcontext
from tempfile import TemporaryDirectory


@main.group()
def utils():
    """
    Data store management utilities.
    """
    pass


@utils.command(parents=[path_arg])
def list(path):
    """
    List figures and data series stored in the data store.
    """
    store = DataStore(path)

    # Data handlers.
    print("Data handlers")
    for _h in store.get_data_handlers():
        print(_h.decoded_data_def)

    # Figure handlers.
    print("Figure handlers.")
    for _h in store.get_figure_handlers():
        print(_h.decoded_data_def)
