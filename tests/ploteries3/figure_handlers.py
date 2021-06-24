from ploteries3 import figure_handlers as mdl
from dash.dependencies import Input, Output, ALL
import numpy.testing as npt
from pglib.py import SSQ
import plotly.graph_objects as go
from ploteries3.ndarray_data_handlers import UniformNDArrayDataHandler
from unittest import TestCase
from .data_store import get_store
import numpy as np
from pglib.profiling import time_and_print
from contextlib import contextmanager
import dash


@contextmanager
def get_store_with_fig():
    with get_store() as store:
        #
        arr1_h = UniformNDArrayDataHandler(store, 'arr1')
        for index in range(5):
            arr1_h.add_data(
                index, np.array(
                    list(zip(range(0, 10), range(10, 20))),
                    dtype=[('f0', 'i'), ('f1', 'f')]))
        #
        arr2_h = UniformNDArrayDataHandler(store, 'arr2')
        for index in range(5):
            arr2_h.add_data(
                index, np.array(
                    list(zip(range(20, 30), range(30, 40))),
                    dtype=[('f0', 'i'), ('f1', 'f')]))

        #
        fig = go.Figure()
        for k in range(2):
            fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name=f'plot_{k}'))

        #
        fig_h = mdl.FigureHandler(
            store,
            'fig1',
            {'source1': 'arr1',
             'source2': 'arr2'},
            [{'figure_keys': ('data', 0, 'x'),
              'source_keys': ('source1', 'data', 0, 'f0')},
             {'figure_keys': SSQ()['data'][0]['y'],
              'source_keys': SSQ()['source1']['data'][0]['f1']},
             #
             (('data', 1, 'x'),
              ('source2', 'data', 0, 'f0')),
             (('data', 1, 'y'),
              ('source2', 'data', 0, 'f1'))],
            figure=fig)

        yield store, arr1_h, arr2_h, fig_h


class TestFigureHandler(TestCase):
    def test_create(self):

        with get_store_with_fig() as (store, arr1_h, arr2_h, fig_h):
            built_fig = fig_h.build_figure()

            npt.assert_array_equal(
                SSQ()['data', 0, 'x'](built_fig),
                arr1_h.load_data()['data'][0]['f0'])

            npt.assert_array_equal(
                SSQ()['data', 0, 'y'](built_fig),
                arr1_h.load_data()['data'][0]['f1'])

            npt.assert_array_equal(
                SSQ()['data', 1, 'x'](built_fig),
                arr2_h.load_data()['data'][0]['f0'])

            npt.assert_array_equal(
                SSQ()['data', 1, 'y'](built_fig),
                arr2_h.load_data()['data'][0]['f1'])

            # Write the definition to the store
            fig_h.write_def()
            fig_h_loaded = mdl.FigureHandler.from_name(store, fig_h.name)

            # Compare figures.
            built_fig_loaded = fig_h_loaded.build_figure()
            self.assertEqual(
                built_fig_json := built_fig.to_json(),
                built_fig_loaded.to_json())

            # Compare html
            self.assertEqual(
                str(html := fig_h.build_html()),
                str(fig_h_loaded.build_html()))

            # Compare figure form html
            self.assertEqual(
                html.children[0].children[1].figure.to_json(),
                built_fig_json)

    def test_encode_decode_params(self):
        with get_store_with_fig() as (store, arr1_h, arr2_h, fig_h):
            orig_params = {
                'sources': dict(fig_h.sources),
                'mappings': tuple(fig_h.mappings),
                'figure': dict(fig_h.figure)}
            mdl.FigureHandler.decode_params(decoded_params := fig_h.encode_params())
            self.assertDictEqual(orig_params, decoded_params)

    def test_create_dash_callbacks(self):
        app = dash.Dash()
        with get_store_with_fig() as (store, arr1_h, arr2_h, fig_h):
            #
            mdl.FigureHandler.create_dash_callbacks(
                app,
                store,
                n_interval_input=Input('interval-component', 'n_intervals'),
                global_index_input_value=Input('global-index-dropdown', 'value'),
                global_index_dropdown_options=Output("global-step-dropdown", "options"),
                registry=(registry := set()))
            #
            self.assertEqual(registry, {mdl.FigureHandler})

    def test_update_all_sliders_and_global_index_dropdown_options_callback(self):

        with get_store_with_fig() as (store, arr1_h, arr2_h, fig_h):

            callback = mdl.FigureHandler._update_all_sliders_and_global_index_dropdown_options

            output = callback(
                data_store=store,
                n_intervals=0,
                global_index=(global_index := 3),
                slider_ids=[
                    {'type': mdl.FigureHandler.encoded_class_name(),
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
