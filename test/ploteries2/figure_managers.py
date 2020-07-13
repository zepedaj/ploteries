from unittest import TestCase
import re
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
            # out = mdl.ScalarsManager(writer).load('scalars1')
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


class TestPlotsManager(TestCase):
    def test_add_plots(self):
        with NamedTemporaryFile() as tmpfo:
            # Create data and figure
            writer = Writer(tmpfo.name)
            mdl.PlotsManager.add_plots(
                writer, 'plots1', [np.arange(6).reshape(2, 3), np.arange(6, 12).reshape(2, 3)], 0)
            writer.flush()

    def test_atomic_creation_of_tables(self):
        with NamedTemporaryFile() as tmpfo:
            # Create table with name that crashes with derived table names.
            writer = Writer(tmpfo.name)
            writer.add_data('plots1__0', np.ndarray([0, 1, 2]), 0)
            try:
                mdl.PlotsManager.add_plots(
                    writer, 'plots1', [np.arange(6).reshape(2, 3), np.arange(6, 12).reshape(2, 3)], 0)
            except sqa.exc.InvalidRequestError as err:
                if not re.match(re.escape('Table ')+'.*'+re.escape(
                        "is already defined for this MetaData instance.  Specify 'extend_existing=True' "
                        "to redefine options and columns on an existing Table object."), err.args[0]):
                    raise
