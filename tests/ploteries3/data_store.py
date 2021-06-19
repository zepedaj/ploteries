from unittest import TestCase
import ploteries3.data_store as mdl
from tempfile import NamedTemporaryFile


class TestDataStore(TestCase):
    def test_create(self):
        with NamedTemporaryFile() as tmpf:
            obj = mdl.DataStore(tmpf.name)
            for tbl_name in ['data_records', 'writers', 'data_defs', 'figure_defs']:
                self.assertIn(tbl_name, obj._metadata.tables.keys())
