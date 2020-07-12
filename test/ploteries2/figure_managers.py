from unittest import TestCase
from ploteries2 import figure_managers as mdl
from ploteries2.writer import Writer
from tempfile import NamedTemporaryFile
import sqlalchemy as sqa
import numpy as np
import numpy.testing as npt
from pglib.py import SliceSequence


class TestScalarsManager(TestCase):
    def test_load(self):
        with NamedTemporaryFile() as tmpfo:
            # Create data and figure
            writer = Writer(tmpfo.name)
            mdl.ScalarsManager.add_scalars(
                writer, 'scalars1', np.array([0, 1, 2]), 0)
            mdl.ScalarsManager.add_scalars(
                writer, 'scalars1', np.array([3, 4, 5]), 1)
            writer.flush()

            # Load and verify.
            #out = mdl.ScalarsManager(writer).load('scalars1')
            out = mdl.load_figure(writer, tag='scalars1')
            #
            self.assertEqual(len(out['data']), 6)
            #
            npt.assert_equal(out['data'][0]['y'], [0, 3])
            npt.assert_equal(out['data'][1]['y'], [1, 4])
            npt.assert_equal(out['data'][2]['y'], [2, 5])
            #
            npt.assert_equal(out['data'][0]['x'], [0, 1])
            npt.assert_equal(out['data'][1]['x'], [0, 1])
            npt.assert_equal(out['data'][2]['x'], [0, 1])

    def test_add_scalars(self):
        with NamedTemporaryFile() as tmpfo:
            writer = Writer(tmpfo.name)
            mdl.ScalarsManager.add_scalars(
                writer, 'scalars1', np.array([0, 1, 2]), 0)
            mdl.ScalarsManager.add_scalars(
                writer, 'scalars1', np.array([3, 4, 5]), 1)
            writer.flush()

            # Verify figure exists.
            dat = writer.execute(sqa.select(
                [writer._figures]).where(sqa.column('tag') == 'scalars1'))
            self.assertEqual(len(dat), 1)

            # Verify data exists.
            dat = writer.execute(sqa.select(
                [writer.get_data_table('scalars1').c.content]))
            self.assertEqual(len(dat), 2)
