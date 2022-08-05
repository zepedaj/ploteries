from ploteries._bin_helpers import launch as mdl
from unittest import TestCase
from ploteries.cli_interface import PloteriesLaunchInterface
from dash.development.base_component import Component
from ..figure_handler import get_store_with_fig


class TestFunctions(TestCase):
    def test_call_methods(self):
        with get_store_with_fig() as (store, arr1_h, arr2_h, fig_h):
            #
            mdl.DATA_INTERFACES = mdl.DataInterfaces(store.path)

            layout = mdl.create_layout(1)
            self.assertIsInstance(layout, Component)

            #
            mdl.create_toolbar_callbacks()
