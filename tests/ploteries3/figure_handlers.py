from ploteries3 import figure_handlers as mdl
import numpy.testing as npt
from pglib.py import SSQ
import plotly.graph_objects as go
from ploteries3.ndarray_data_handlers import UniformNDArrayDataHandler
from unittest import TestCase
from .data_store import get_store
import numpy as np
from pglib.profiling import time_and_print


class TestFigureHandler(TestCase):
    def test_create(self):
        with get_store() as store:
            #
            arr1_dh = UniformNDArrayDataHandler(store, 'arr1')
            arr1_dh.add_data(
                0, np.array(
                    list(zip(range(0, 10), range(10, 20))),
                    dtype=[('f0', 'i'), ('f1', 'f')]))
            #
            arr2_dh = UniformNDArrayDataHandler(store, 'arr2')
            arr2_dh.add_data(
                0, np.array(
                    list(zip(range(20, 30), range(30, 40))),
                    dtype=[('f0', 'i'), ('f1', 'f')]))

            #
            fig = go.Figure()
            for k in range(2):
                fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name=f'plot_{k}'))

            #
            fh = mdl.FigureHandler(
                store,
                'fig1',
                {'source1': 'arr1',
                 'source2': 'arr2'},
                {('data', 0, 'x'): ('source1', 'data', 0, 'f0'),
                 ('data', 0, 'y'): ('source1', 'data', 0, 'f1'),
                 #
                 ('data', 1, 'x'): ('source2', 'data', 0, 'f0'),
                 ('data', 1, 'y'): ('source2', 'data', 0, 'f1')},
                figure=fig)

            built_fig = fh.build_figure()

            npt.assert_array_equal(
                SSQ()['data', 0, 'x'](built_fig),
                arr1_dh.load_data()['data'][0]['f0'])

            npt.assert_array_equal(
                SSQ()['data', 0, 'y'](built_fig),
                arr1_dh.load_data()['data'][0]['f1'])

            npt.assert_array_equal(
                SSQ()['data', 1, 'x'](built_fig),
                arr2_dh.load_data()['data'][0]['f0'])

            npt.assert_array_equal(
                SSQ()['data', 1, 'y'](built_fig),
                arr2_dh.load_data()['data'][0]['f1'])
