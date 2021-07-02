from unittest import TestCase
import numpy.testing as npt
from dash.dependencies import Input, Output, ALL
import dash
from .figure_handler import get_store_with_fig
from ploteries3 import _cli_interface as mdl
from ploteries3.data_store import Col_


class TestPloteriesLaunchInterface(TestCase):

    def test_render_empty_figures(self):

        with get_store_with_fig() as (store, arr1_h, arr2_h, fig_h):
            built_fig = fig_h.build_figure()

            # Write the definition to the store
            fig_h.write_def()
            pli = mdl.PloteriesLaunchInterface(
                store)

            #
            empty_figs = pli.render_empty_figures()
            self.assertEqual(len(empty_figs), 1)
            self.assertEqual(empty_figs[0].name, 'fig1')

            # Compare traces from pli to traces from figure handler.
            fig_from_handle_traces = store.get_figure_handlers(
                Col_('name') == 'fig1')[0].build_figure().to_dict()['data']
            fig_from_pli_traces = pli._build_formatted_figure_from_name('fig1').to_dict()['data']
            self.assertEqual(len(fig_from_handle_traces), len(fig_from_pli_traces))
            for trace1, trace2 in zip(fig_from_handle_traces, fig_from_pli_traces):
                for key in set(trace1.keys()).union(trace2.keys()):
                    npt.assert_array_equal(
                        trace1[key],
                        trace2[key])

    def test_create_callbacks(self):
        app = dash.Dash()
        with get_store_with_fig() as (store, arr1_h, arr2_h, fig_h):
            #
            mdl.PloteriesLaunchInterface(store).create_callbacks(
                app,
                n_interval_input=Input('interval-component', 'n_intervals'),
                global_index_input_value=Input('global-index-dropdown', 'value'),
                global_index_dropdown_options=Output("global-step-dropdown", "options"))

    def test_update_all_sliders_and_global_index_dropdown_options_callback(self):

        with get_store_with_fig() as (store, arr1_h, arr2_h, fig_h):

            output = mdl.PloteriesLaunchInterface(
                store)._update_all_sliders_and_global_index_dropdown_options(
                    n_intervals=0,
                    global_index=(global_index := 3),
                    slider_ids=[
                        {'type': mdl.PloteriesLaunchInterface.encoded_class_name(),
                         'element': 'slider',
                         'name': ALL}])

            # 'marks', 'min', 'max', 'value', 'disabled'
            self.assertEqual(
                output,
                [
                    [{0: '0', 1: '', 2: '', 3: '', 4: '4'}],
                    [0],
                    [4],
                    [global_index],
                    [False],
                    [{'label': '0', 'value': 0},
                     {'label': '1', 'value': 1},
                     {'label': '2', 'value': 2},
                     {'label': '3', 'value': 3},
                     {'label': '4', 'value': 4}]
                ])
