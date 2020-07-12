from unittest import TestCase
from ploteries2.writer import Writer
from ploteries2 import reader as mdl
from ploteries2.figure_managers import ScalarsManager
from tempfile import NamedTemporaryFile
import sqlalchemy as sqa
import numpy as np
import numpy.testing as npt
from pglib.py import SliceSequence
from plotly import graph_objects as go


class TestReader(TestCase):
    def test_get_figures(self):
        with NamedTemporaryFile() as tmpfo:
            # Create scalars
            writer = Writer(tmpfo.name)
            writer.add_scalars('scalars1', np.array([0, 1, 2]), 0)
            writer.add_scalars('scalars2', np.array([3, 4, 5]), 1)
            writer.flush()

            #
            reader = mdl.Reader(tmpfo.name)
            figure_recs = reader.get_figure_recs()
            self.assertEqual(
                set([x.tag for x in figure_recs]), {'scalars1', 'scalars2'})
            #
            self.assertEqual(set(figure_recs[0].keys()), {
                             'id', 'tag', 'figure', 'manager', 'name'})
            self.assertEqual(set([x.name for x in figure_recs]), {
                             'fig_1', 'fig_2'})

            # Filter by manager
            reader = mdl.Reader(tmpfo.name)
            figure_recs = reader.get_figure_recs(manager=ScalarsManager)

            self.assertEqual(
                set([x.tag for x in figure_recs]), {'scalars1', 'scalars2'})

            # Filter by id
            reader = mdl.Reader(tmpfo.name)
            figure_recs = reader.get_figure_recs(id=1)

            self.assertEqual(
                set([x.tag for x in figure_recs]), {'scalars1'})

            # Filter by name
            reader = mdl.Reader(tmpfo.name)
            figure_recs = reader.get_figure_recs(name='fig_1')

            self.assertEqual(
                set([x.tag for x in figure_recs]), {'scalars1'})

            # Filter by tag
            reader = mdl.Reader(tmpfo.name)
            figure_recs = reader.get_figure_recs(tag='scalars1')

            self.assertEqual(
                set([x.name for x in figure_recs]), {'fig_1'})
