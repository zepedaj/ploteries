from unittest import TestCase
import re
from ploteries2 import figure_managers as mdl
from ploteries2.writer import Writer
from tempfile import NamedTemporaryFile
import sqlalchemy as sqa
import numpy as np
import numpy.testing as npt
from pglib.py import SliceSequence


class TestFunctions(TestCase):
    def test_global_steps(self):
        with NamedTemporaryFile() as tmpfo:
            writer = Writer(tmpfo.name)
            mdl.ScalarsManager.add_scalars(
                writer, 'scalars1', np.array([0, 1, 2]), 0)
            mdl.ScalarsManager.add_scalars(
                writer, 'scalars1', np.array([3, 4, 5]), 1)
            writer.flush()

            # Check global steps
            self.assertEqual(mdl.global_steps(
                writer, tag='scalars1'), [0, 1])


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
            #

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

    def test_load(self):
        with NamedTemporaryFile() as tmpfo:
            # Create data and figure
            writer = Writer(tmpfo.name)
            dat00, dat01 = np.random.randn(2, 10), np.random.randn(2, 7)
            dat10, dat11 = np.random.randn(2, 6), np.random.randn(2, 11)
            mdl.PlotsManager.add_plots(
                writer, 'plots1', [dat00, dat01], 0)
            mdl.PlotsManager.add_plots(
                writer, 'plots1', [dat10, dat11], 1)
            writer.flush()

            #
            pm = mdl.PlotsManager(writer, limit=None)
            dat = pm.load_data(1, as_np_arrays=False)

            # Load and verify.
            out = mdl.load_figure(writer, tag='plots1')
            # #
            self.assertEqual(len(out['data']), 2)
            # #
            npt.assert_equal(out['data'][0]['x'], dat00[0])
            npt.assert_equal(out['data'][0]['y'], dat00[1])
            # npt.assert_equal(out['data'][1]['y'], [1, 4])
            # npt.assert_equal(out['data'][2]['y'], [2, 5])
            # #
            # npt.assert_equal(out['data'][0]['x'], [0, 1])
            # npt.assert_equal(out['data'][1]['x'], [0, 1])
            # npt.assert_equal(out['data'][2]['x'], [0, 1])


class TestHistogramsManager(TestCase):
    def test_add_histograms(self):
        with NamedTemporaryFile() as tmpfo:
            # Create data and figure
            writer = Writer(tmpfo.name)
            mdl.HistogramsManager.add_histograms(
                writer, 'histo1', [np.arange(6), np.arange(6, 12)], 0)
            writer.flush()
