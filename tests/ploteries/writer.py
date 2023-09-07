import ploteries.writer as mdl
import numpy.testing as npt
from unittest import TestCase
from tempfile import NamedTemporaryFile
from ploteries.data_store import DataStore
import numpy as np


class TestWriter(TestCase):
    def test_add_scalars(self):
        with NamedTemporaryFile() as tmp_fo:
            # Write data
            writer = mdl.Writer(tmp_fo.name)
            num_traces = 3
            figure_name = "fig1"
            writer.add_scalars(figure_name, rec0_arr := np.array([0] * num_traces), 0)
            writer.add_scalars(figure_name, rec1_arr := np.array([1] * num_traces), 1)

            data_name = mdl.Writer._get_table_name(
                "add_scalars", figure_name=figure_name
            )

            # Verify contents.
            store = DataStore(tmp_fo.name)
            self.assertEqual(
                [_x.name for _x in store.get_figure_handlers()], [figure_name]
            )
            self.assertEqual([_x.name for _x in store.get_data_handlers()], [data_name])

            # Check loaded data.
            fig_h = store.get_figure_handlers()[0]

            # Check built figure.
            fig = fig_h.build_figure()
            self.assertEqual(len(fig["data"]), num_traces * 2)

            for _k in range(num_traces):
                npt.assert_array_equal(
                    np.stack((rec0_arr, rec1_arr))[:, _k], fig["data"][_k]["y"]
                )

    def test_add_plots(self):
        with NamedTemporaryFile() as tmp_fo:
            # Write data
            writer = mdl.Writer(tmp_fo.name)
            figure_name = "fig1"
            writer.add_plots(
                figure_name,
                traces := [
                    {"x": np.arange(0, 10), "y": np.arange(10, 20), "text": ["a"] * 10},
                    {
                        "x": np.arange(20, 30),
                        "y": np.arange(30, 40),
                        "text": ["b"] * 10,
                    },
                ],
                0,
            )

            data_name = mdl.Writer._get_table_name("add_plots", figure_name=figure_name)

            # Verify contents.
            store = DataStore(tmp_fo.name)
            self.assertEqual(
                [_x.name for _x in store.get_figure_handlers()], [figure_name]
            )
            self.assertEqual([_x.name for _x in store.get_data_handlers()], [data_name])

            # Check loaded data.
            fig_h = store.get_figure_handlers()[0]

            # Check built figure.
            fig = fig_h.build_figure()
            default_trace_kwargs = {"type": "scatter", "mode": "lines"}
            [_x.update(**default_trace_kwargs) for _x in traces]
            npt.assert_equal(fig.to_dict()["data"], traces)

    def test_add_table(self):
        with NamedTemporaryFile() as tmp_fo:
            writer = mdl.Writer(tmp_fo.name)
            table_name = "table1"
            tbl = writer.add_table(table_name, values := {"Col 1": 1, "Col 2": 2}, 0)

            data_name = mdl.Writer._get_table_name("add_table", figure_name=table_name)

            # Verify contents.
            store = DataStore(tmp_fo.name)
            self.assertEqual(
                [_x.name for _x in store.get_figure_handlers()], [table_name]
            )
            self.assertEqual([_x.name for _x in store.get_data_handlers()], [data_name])

            # Verify data.
            store.flush()
            self.assertEqual(store[data_name]["data"], [values])
