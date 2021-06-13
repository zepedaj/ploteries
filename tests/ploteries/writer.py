from unittest import TestCase
from tempfile import TemporaryDirectory
import os.path as osp
import numpy as np
#
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, ForeignKey, select
from ploteries import writer as ptw
#
import plotly.express as px, plotly.graph_objects as go


class TestWriter(TestCase):
    def _helper_figure(self):
        tips = px.data.tips()
        fig = px.histogram(tips, x="total_bill")        
        return fig

    def test_create(self):
        with TemporaryDirectory() as td:
            # Create 
            writer=ptw.Writer(td)
            writer._engine.table_names()
            self.assertTrue(osp.isfile(writer.path))
            
            # Add one figure
            figure1 = self._helper_figure()
            writer.add_figure('abc', figure1, 1)
            writer.flush()
            self.assertEqual(writer._engine.table_names(), ['__ps__table_meta', 'abc'])
            #
            result = list(writer._execute(select([writer._metadata.tables['abc']])))
            self.assertEqual(len(result), 1)
            
            # Add second figure
            figure2 = self._helper_figure()
            writer.add_figure('abc', figure2,2)
            #
            writer.flush()
            result = list(writer._execute(select([writer._metadata.tables['abc']])))            
            self.assertEqual(len(result), 2)
            
            # Add 3,4 figures
            figure3 = self._helper_figure()
            writer.add_figure('xyz', figure3,2)
            writer.add_figure('fig3', figure3,2)
            writer.add_figure('fig4', figure3,2)
            
            # Test field is plotly figure
            self.assertEqual(type(result[0]['content']), go.Figure)
            
            # Add scalar
            writer.add_scalar('scalar1',  1, 1)
            writer.add_scalar('scalar1',  2, 2)
            
            # Add scalar
            writer.add_scalar('scalar2',  5, 1)
            writer.add_scalar('scalar2',  -1, 2)

            # Add scalar
            writer.add_scalar('scalar3',  5, 1)
            writer.add_scalar('scalar3',  -1, 2)
            writer.add_scalar('scalar4',  5, 1)
            writer.add_scalar('scalar4',  -1, 2)

            # Add histogram
            writer.add_histogram('histoA' , np.random.randn(1000), 1)

            writer.flush()
            #import ipdb; ipdb.set_trace()
