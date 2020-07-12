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
    def test_load_figures(self):
        with NamedTemporaryFile() as tmpfo:
            # Create scalars
            writer = Writer(tmpfo.name)
            writer.add_scalars('scalars1', np.array([0, 1, 2]), 0)
            writer.add_scalars('scalars1', np.array([0, 2, 3]), 1)
            writer.add_scalars('scalars2', np.array([3, 4, 5]), 1)
            writer.flush()

            # Filter by manager
            figure_recs = writer.load_figure_recs(manager=ScalarsManager)
            fig_objs = [writer.load_figure(_f) for _f in figure_recs]
            self.assertEqual(
                list(map(lambda x: x.data[0].y.shape, fig_objs)), [(2,), (1,)])  # Order guaranteed?

            #
            reader = mdl.Reader(tmpfo.name)
            figure_recs = reader.load_figure_recs()
            self.assertEqual(
                set([x.tag for x in figure_recs]), {'scalars1', 'scalars2'})
            #
            self.assertEqual(set(figure_recs[0].keys()), {
                             'id', 'tag', 'figure', 'manager', 'name'})
            self.assertEqual(set([x.name for x in figure_recs]), {
                             'fig_1', 'fig_2'})

            # Filter by manager
            figure_recs = reader.load_figure_recs(manager=ScalarsManager)
            fig_obj = [reader.load_figure(_f) for _f in figure_recs]
            self.assertEqual(
                set([x.tag for x in figure_recs]), {'scalars1', 'scalars2'})

            # Filter by id
            figure_recs = reader.load_figure_recs(id=1)
            fig_obj = [reader.load_figure(_f) for _f in figure_recs]
            self.assertEqual(
                set([x.tag for x in figure_recs]), {'scalars1'})

            # Filter by name
            figure_recs = reader.load_figure_recs(name='fig_1')
            fig_obj = [reader.load_figure(_f) for _f in figure_recs]
            self.assertEqual(
                set([x.tag for x in figure_recs]), {'scalars1'})

            # Filter by tag
            figure_recs = reader.load_figure_recs(tag='scalars1')
            fig_obj = [reader.load_figure(_f) for _f in figure_recs]
            self.assertEqual(
                set([x.name for x in figure_recs]), {'fig_1'})
