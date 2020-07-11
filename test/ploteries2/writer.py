from unittest import TestCase
from ploteries2 import writer as mdl
from tempfile import NamedTemporaryFile
import sqlalchemy as sqa
import numpy as np
import numpy.testing as npt
from pglib.py import SliceSequence
from plotly import graph_objects as go


class TestWriter(TestCase):
    def test_create_table_and_add_data(self):
        with NamedTemporaryFile() as tmpfo:
            writer = mdl.Writer(tmpfo.name)
            writer._create_table('np_data1', np.ndarray)
            arr = np.array([0, 1, 2, 3])
            writer.add_data('np_data1', arr, 0)
            writer.flush()

            with writer.engine.begin() as conn:
                out = conn.execute(sqa.select(
                    [writer.get_table('np_data1').c.content])).fetchall()
            npt.assert_equal(out[0][0], arr)

    def test_register_display(self):
        with NamedTemporaryFile() as tmpfo:
            writer = mdl.Writer(tmpfo.name)
            writer._create_table('np_data1', np.ndarray)

            figure = go.Figure()
            #
            np_data1 = writer.get_table('np_data1')

            # Check 0
            out = writer.execute(sqa.select(
                [sqa.func.count(writer._figures).label('count')]))
            self.assertEqual(out[0]['count'], 0)
            out = writer.execute(
                sqa.select([sqa.func.count(writer._data_templates).label('count')]))
            self.assertEqual(out[0]['count'], 0)

            # Fail registration of new display, check  no inconsistent state
            with self.assertRaises(sqa.exc.StatementError):
                writer.register_display(
                    'plots/figure1', figure,
                    ({'abc'}, {'x': ('content', None)}))  # TODO
            out = writer.execute(
                sqa.select([sqa.func.count(writer._figures).label('count')]))
            self.assertEqual(out[0]['count'], 0)
            out = writer.execute(
                sqa.select([sqa.func.count(writer._data_templates).label('count')]))
            self.assertEqual(out[0]['count'], 0)

            # Successfully register new display.
            orig_sql = sqa.select([np_data1.c.content, np_data1.c.global_step])
            writer.register_display(
                'plots/figure2', figure,
                (orig_sql, [(['data', 0, 'x'], ['content'])]))

            # Check insertions
            out = writer.execute(
                sqa.select([sqa.func.count(writer._figures).label('count')]))
            self.assertEqual(out[0]['count'], 1)
            out = writer.execute(
                sqa.select([sqa.func.count(writer._data_templates).label('count')]))
            self.assertEqual(out[0]['count'], 1)

            retrieved = writer.execute(
                sqa.select([writer._data_templates]))[0]

            # Check retrieved query matches original
            self.assertEqual(str(retrieved['sql']), str(orig_sql))
