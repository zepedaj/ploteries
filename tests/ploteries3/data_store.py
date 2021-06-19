from unittest import TestCase
import ploteries3.data_store as mdl
from tempfile import NamedTemporaryFile
from contextlib import contextmanager


@contextmanager
def get_store():
    with NamedTemporaryFile() as tmpf:
        obj = mdl.DataStore(tmpf.name)
        yield obj


class TestDataStore(TestCase):
    def test_create(self):
        with get_store() as obj:
            for tbl_name in ['data_records', 'writers', 'data_defs', 'figure_defs']:
                self.assertIn(tbl_name, obj._metadata.tables.keys())
