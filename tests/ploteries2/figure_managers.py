from unittest import TestCase
import re
from ploteries2 import figure_managers as mdl
from ploteries2.writer import Writer
from tempfile import NamedTemporaryFile
import sqlalchemy as sqa
import numpy as np
import numpy.testing as npt
from pglib.py import SliceSequence
import warnings


def _xy(arrays):
    return [dict(zip('xy', _arr)) for _arr in arrays]


class TestFunctions(TestCase):
    def test_global_steps(self):
        with NamedTemporaryFile() as tmpfo:
            writer = Writer(tmpfo.name)
            mdl.SmoothenedScalarsManager.add_scalars(
                writer, 'scalars1', np.array([0, 1, 2]), 0)
            mdl.SmoothenedScalarsManager.add_scalars(
                writer, 'scalars1', np.array([3, 4, 5]), 1)
            writer.flush()

            # Check global steps
            self.assertEqual(mdl.global_steps(
                writer, tag='scalars1'), [0, 1])


class TestSmoothenedScalarsManager(TestCase):
    def test_load(self):
        with NamedTemporaryFile() as tmpfo:
            # Create data and figure
            writer = Writer(tmpfo.name)
            mdl.SmoothenedScalarsManager.add_scalars(
                writer, 'scalars1', np.array([0, 1, 2]), 0)
            mdl.SmoothenedScalarsManager.add_scalars(
                writer, 'scalars1', np.array([3, 4, 5]), 1)
            writer.flush()

            # Load and verify.
            # out = mdl.SmoothenedScalarsManager(writer).load('scalars1')
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
            mdl.SmoothenedScalarsManager.add_scalars(
                writer, 'scalars1', np.array([0, 1, 2]), 0)
            mdl.SmoothenedScalarsManager.add_scalars(
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
    def test_data_subfield(self):
        with NamedTemporaryFile() as tmpfo:
            # Create data and figure
            x, y, err_y = (np.random.randn(5) for _ in (1, 2, 3))
            writer = Writer(tmpfo.name)
            mdl.PlotsManager.add_plots(
                writer, 'plots1',
                [{
                    'x': x,
                    'y': y,
                    'error_y/array': err_y,
                }], 0)
            writer.flush()
            fig = mdl.load_figure(writer, tag='plots1')
            npt.assert_equal(fig['data'][0]['x'], x)
            npt.assert_equal(fig['data'][0]['y'], y)
            npt.assert_equal(fig['data'][0]['error_y']['array'], err_y)

    def test_add_plots(self):
        with NamedTemporaryFile() as tmpfo:
            # Create data and figure
            writer = Writer(tmpfo.name)
            mdl.PlotsManager.add_plots(
                writer, 'plots1',
                _xy([np.arange(6).reshape(2, 3), np.arange(6, 12).reshape(2, 3)]), 0)
            writer.flush()

    def test_atomic_creation_of_tables__derived_table_exists(self):
        with NamedTemporaryFile() as tmpfo:
            # Create table with name that crashes with derived table names.
            writer = Writer(tmpfo.name)
            writer.add_data('plots1__0', {'content': np.ndarray([0, 1, 2])}, 0)
            try:
                mdl.PlotsManager.add_plots(writer, 'plots1', _xy(
                    [np.arange(6).reshape(2, 3), np.arange(6, 12).reshape(2, 3)]), 0)
                raise Exception('Expected exception!')
            except sqa.exc.InvalidRequestError as err:
                if not re.match(
                        re.escape('Table ') + '.*' + re.escape(
                            "is already defined for this MetaData instance.  Specify 'extend_existing=True' "
                            "to redefine options and columns on an existing Table object."),
                        err.args[0]):
                    raise
            # Check no records were created
            self.assertEqual(len(writer.load_figure_recs(tag='plots1')), 0)
            # First table exists.
            writer.get_data_table('plots1__0')
            # Second table does not exist.
            with self.assertRaises(KeyError):
                writer.get_data_table('plots1__1')

    def test_atomic_creation_of_tables__figure_exists(self):

        with NamedTemporaryFile() as tmpfo:

            # Create scalar figure with name that crashes with derived table names.
            writer = Writer(tmpfo.name)
            writer.add_scalars('plots1', np.array([0, 1, 2]), 0)
            if not writer.figure_exists('plots1', {'manager': mdl.SmoothenedScalarsManager}):
                raise Exception('Unexpected error.')
            writer.get_data_table('plots1')  # Ensure data table exists.

            # Attempt to create plots figure
            try:
                mdl.PlotsManager.add_plots(
                    writer, 'plots1',
                    _xy([np.arange(6).reshape(2, 3), np.arange(6, 12).reshape(2, 3)]), 0)
                raise Exception('Expected exception!')
            except Exception as err:  # sqa.exc.InvalidRequestError
                if not re.match(
                        re.escape(
                            "Retrieved figure record (1, 'plots1', <class 'ploteries2."
                            "figure_managers.SmoothenedScalarsManager'>,") +
                        '.+',  # does not match expected values .+',
                        err.args[0]):
                    # if err.args[0] != "Retrieved figure record (1, 'plots1', <class 'ploteries2.figure_managers.SmoothenedScalarsManager'>, Figure({\n    'data': [{'line': {'color': 'hsl(236, 94%, 91%)'},\n              'mode': 'lines',\n              'showlegend': False,\n              'type' ... (950 characters truncated) ...             'showlegend': False,\n              'type': 'scatter',\n              'x': [],\n              'y': []}],\n    'layout': {'template': '...'}\n})) does not match expected values {'manager': <class 'ploteries2.figure_managers.PlotsManager'>}.":
                    import ipdb
                    ipdb.set_trace()
                    raise

            # Check no records were created
            self.assertEqual(len(writer.load_figure_recs(tag='plots1')), 1)

            # Check content type of original table did not change.
            if not writer.figure_exists('plots1', {'manager': mdl.SmoothenedScalarsManager}):
                raise Exception('Unexpected error.')

            # Check both derived tables do not exist.
            with self.assertRaises(KeyError):
                writer.get_data_table('plots1__0')
            with self.assertRaises(KeyError):
                writer.get_data_table('plots1__1')

            # Avoids __del__ flush exception due to missing tmp file.
            writer.flush()

    def test_load(self):
        with NamedTemporaryFile() as tmpfo:
            # Create data and figure
            writer = Writer(tmpfo.name)
            dat00, dat01 = np.random.randn(2, 10), np.random.randn(2, 7)
            dat10, dat11 = np.random.randn(2, 6), np.random.randn(2, 11)
            mdl.PlotsManager.add_plots(
                writer, 'plots1',
                _xy([dat00, dat01]), 0)
            mdl.PlotsManager.add_plots(
                writer, 'plots1', _xy([dat10, dat11]), 1)
            writer.flush()

            #
            pm = mdl.PlotsManager(writer, limit=None)
            dat = pm.load_data(1, as_np_arrays=False)

            # Load and verify - should get the latest bar.
            fig = mdl.load_figure(writer, tag='plots1')
            #
            self.assertEqual(len(fig['data']), 2)
            #
            npt.assert_equal(fig['data'][1]['x'], dat11[0])
            npt.assert_equal(fig['data'][1]['y'], dat11[1])
            # npt.assert_equal(fig['data'][1]['y'], [1, 4])
            # npt.assert_equal(fig['data'][2]['y'], [2, 5])
            # #
            # npt.assert_equal(fig['data'][0]['x'], [0, 1])
            # npt.assert_equal(fig['data'][1]['x'], [0, 1])
            # npt.assert_equal(fig['data'][2]['x'], [0, 1])

    def test_register_figure_atomicity(self):
        if True:
            warnings.warn(
                'Skipping current test due to lack of table creation rollback support in sqlite3.')
            return
        with NamedTemporaryFile() as tmpfo:
            # Create data and figure
            writer = Writer(tmpfo.name)
            try:
                mdl.PlotsManager.register_figure(
                    writer, 'plots1', 3, _test_exceptions={'post_create_tables'})  # , 'pre_sql'})
                raise Exception('Unraised exception.')
            except Exception as err:
                if err.args[0] != 'post_create_tables':  # 'pre_sql':
                    raise
                if writer.figure_exists('plots1'):
                    raise Exception('Figure not expected.')
                for k in range(3):
                    table_name = f'plots1__{k}'
                    with self.assertRaises(KeyError, msg=f'Table {table_name} not expected.'):
                        writer.get_data_table(table_name)
                pass


class TestHistogramsManager(TestCase):
    def test_add_histograms(self):
        with NamedTemporaryFile() as tmpfo:
            # Create data and figure
            writer = Writer(tmpfo.name)
            mdl.HistogramsManager.add_histograms(
                writer, 'histo1', [np.arange(6), np.arange(6, 12)], 0)
            writer.flush()
