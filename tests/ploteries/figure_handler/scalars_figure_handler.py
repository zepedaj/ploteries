from unittest import TestCase
from ploteries.figure_handler import scalars_figure_handler as mdl
from numpy import testing as npt
from .figure_handler import TestFigureHandler


class TestScalarsFigureHandler(TestFigureHandler):
    FigureHandler = mdl.ScalarsFigureHandler

    def test_create(self):
        with self.get_store_with_fig() as (store, arr1_h, arr2_h, fig_h):
            assert isinstance(fig_h, mdl.ScalarsFigureHandler)

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
