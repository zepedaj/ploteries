from unittest import TestCase
import ploteries3.data_store as mdl
from tempfile import NamedTemporaryFile
from contextlib import contextmanager
import pglib.numpy as pgnp
import numpy as np
import numpy.testing as npt
from sqlalchemy.sql import column as c
from ploteries3.ndarray_data_handlers import UniformNDArrayDataHandler
from pglib.sqlalchemy import ClassType
from sqlalchemy.sql.expression import bindparam


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

    def test_get_data_handlers(self):
        num_arrays = 10
        with get_store() as store:
            dh = UniformNDArrayDataHandler(store, 'arr1')

            arrs = [pgnp.random_array((10, 5, 7), dtype=[('f0', 'datetime64'), ('f1', 'int')])
                    for _ in range(num_arrays)]

            [dh.add_data(0, _arr) for _arr in arrs]

            dat = dh.load_data()
            npt.assert_array_equal(dat['data'], np.array(arrs))

            for dh in [
                store.get_data_handlers(c('name') == 'arr1')[0],
                # store.get_data_handlers(c('handler') == UniformNDArrayDataHandler)[0],# Not working.
            ]:
                npt.assert_array_equal(
                    dh.load_data()['data'],
                    np.array(arrs))
