import ploteries3.writer as mdl
import numpy.testing as npt
from unittest import TestCase
from tempfile import NamedTemporaryFile
from ploteries3.data_store import DataStore
import numpy as np


class TestWriter(TestCase):
    def test_add_scalars(self):
        with NamedTemporaryFile() as tmp_fo:
            # Write data
            writer = mdl.Writer(tmp_fo.name)
            num_traces = 3
            fig_name = 'fig1'
            writer.add_scalars(fig_name, rec0_arr := np.array([0]*num_traces), 0)
            writer.add_scalars(fig_name, rec1_arr := np.array([1]*num_traces), 1)

            data_name = mdl.Writer._get_table_name('add_scalars', tag=fig_name)

            # Verify contents.
            store = DataStore(tmp_fo.name)
            self.assertEqual([_x.name for _x in store.get_figure_handlers()],
                             [fig_name])
            self.assertEqual([_x.name for _x in store.get_data_handlers()],
                             [data_name])

            # Check loaded data.
            fig_h = store.get_figure_handlers()[0]

            # Check built figure.
            fig = fig_h.build_figure()
            self.assertEqual(len(fig['data']), num_traces)

            for _k in range(num_traces):
                npt.assert_array_equal(
                    np.stack((rec0_arr, rec1_arr))[:, _k],
                    fig['data'][_k]['y'])
