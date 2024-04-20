from ploteries.figure_handler import figure_handler as mdl
from ploteries.data_store import Ref_
import numpy.testing as npt
from jztools.slice_sequence import SSQ_
import plotly.graph_objects as go
from ploteries.ndarray_data_handlers import UniformNDArrayDataHandler
from unittest import TestCase
from ..data_store import get_store
import numpy as np
from jztools.profiling import time_and_print
from contextlib import contextmanager
import dash


@contextmanager
def get_store_with_fig():
    with TestFigureHandler.get_store_with_fig() as out:
        yield out


class TestFigureHandler(TestCase):
    FigureHandler = mdl.FigureHandler

    @classmethod
    @contextmanager
    def get_store_with_fig(cls):
        with get_store() as store:
            #
            arr1_h = UniformNDArrayDataHandler(store, "arr1")
            for index in range(5):
                arr1_h.add_data(
                    index,
                    np.array(
                        list(zip(range(0, 10), range(10, 20))),
                        dtype=[("f0", "i"), ("f1", "f")],
                    ),
                )
            #
            arr2_h = UniformNDArrayDataHandler(store, "arr2")
            for index in range(5):
                arr2_h.add_data(
                    index,
                    np.array(
                        list(zip(range(20, 30), range(30, 40))),
                        dtype=[("f0", "i"), ("f1", "f")],
                    ),
                )

            #
            fig = go.Figure()
            for k in range(2):
                fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name=f"plot_{k}"))
            fig_dict = fig.to_dict()
            fig_dict["data"][0]["x"] = Ref_("arr1")["data"][0]["f0"]
            fig_dict["data"][0]["y"] = Ref_("arr1")["data"][0]["f1"]
            fig_dict["data"][1]["x"] = Ref_("arr2")["data"][0]["f0"]
            fig_dict["data"][1]["y"] = Ref_("arr2")["data"][0]["f1"]

            #
            fig_h = cls.FigureHandler(store, "fig1", fig_dict)

            store.flush()
            yield store, arr1_h, arr2_h, fig_h
            store.flush()

    def test_create(self):
        with get_store_with_fig() as (store, arr1_h, arr2_h, fig_h):
            built_fig = fig_h.build_figure()

            npt.assert_array_equal(
                built_fig["data"][0]["x"], arr1_h.load_data()["data"][0]["f0"]
            )

            npt.assert_array_equal(
                built_fig["data"][0]["y"], arr1_h.load_data()["data"][0]["f1"]
            )

            npt.assert_array_equal(
                built_fig["data"][1]["x"], arr2_h.load_data()["data"][0]["f0"]
            )

            npt.assert_array_equal(
                built_fig["data"][1]["y"], arr2_h.load_data()["data"][0]["f1"]
            )

            # Write the definition to the store
            fig_h.write_def()
            fig_h_loaded = mdl.FigureHandler.from_name(store, fig_h.name)

            # Compare figures.
            built_fig_loaded = fig_h_loaded.build_figure()
            self.assertEqual(
                built_fig_json := built_fig.to_json(), built_fig_loaded.to_json()
            )

    def test_encode_decode_params(self):
        with get_store_with_fig() as (store, arr1_h, arr2_h, fig_h):
            orig_params = {"figure_dict": dict(fig_h.figure_dict)}
            mdl.FigureHandler.decode_params(decoded_params := fig_h.encode_params())
            self.assertDictEqual(orig_params, decoded_params)

    def test_from_traces(self):
        with get_store_with_fig() as (store, arr1_h, arr2_h, fig_h):
            built_fig = fig_h.build_figure()
            fig_h_ft = mdl.FigureHandler.from_traces(
                store, "from_traces", fig_h.figure_dict["data"]
            )
            built_fig_ft = fig_h_ft.build_figure()

            self.assertEqual(built_fig, built_fig_ft)

    def test_get_data_names(self):
        with get_store_with_fig() as (store, arr1_h, arr2_h, fig1_h):
            # Add another fig.
            arr3_h = UniformNDArrayDataHandler(store, "arr3")
            arr3_h.add_data(
                0,
                np.array(
                    list(zip(range(20, 30), range(30, 40))),
                    dtype=[("f0", "i"), ("f1", "f")],
                ),
            )

            fig2_h = mdl.FigureHandler.from_traces(
                store,
                "fig2",
                [{"x": Ref_("arr3")["data"]["f0"], "y": Ref_("arr3")["data"]["f0"]}],
            )

            #
            self.assertEqual(set(fig1_h.get_data_names()), {"arr1", "arr2"})
            self.assertEqual(set(fig2_h.get_data_names()), {"arr3"})

    def test_update(self):
        with get_store_with_fig() as (store, arr1_h, arr2_h, fig_h):
            # fig_h.figure_dict['layout']['template']=None
            with self.assertRaisesRegex(
                Exception,
                r"Cannot update a definition that has not been retrieved from the data store.",
            ):
                fig_h.write_def(mode="update")
            self.assertTrue(fig_h.write_def())
            self.assertFalse(fig_h.write_def())

            fig_h = store.get_figure_handlers()[0]
            self.assertIsInstance(fig_h.figure_dict["layout"]["template"], dict)
            fig_h.figure_dict["layout"]["template"] = None
            fig_h.write_def(mode="update")
            fig_h = store.get_figure_handlers()[0]
            self.assertIsNone(fig_h.figure_dict["layout"]["template"])
