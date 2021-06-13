from unittest import TestCase
from tempfile import TemporaryDirectory
import os.path as osp
#
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, ForeignKey, select
from ploteries import reader as ptr, writer as ptw
#
import plotly.express as px, plotly.graph_objects as go


class TestReader(TestCase):
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
            
            # Add one series
            figure1 = self._helper_figure()
            writer.add_figure('abc', figure1, 1)
            
            # Read figures
            reader=ptr.Reader(writer.path)
            figures = reader.select('abc', 1)
            figures = list(figures)
            self.assertEqual(len(figures), 1)            

            # Add another plot to the same series
            writer.add_figure('abc', figure1, 1)
            figures = reader.select('abc', 1)
            figures = list(figures)
            self.assertEqual(len(figures), 2)
            
            # Add another series
            figure2 = self._helper_figure()
            writer.add_figure('xyz', figure2,1)
            writer.add_figure('xyz', figure2,2)
            
            # Get figure tables
            reader.reflect()
            self.assertEqual(len(reader.tables_of_type('FigureType')), 2)

            # Get global steps
            gs = reader.global_steps('xyz')
            self.assertEqual(gs, [1,2])
