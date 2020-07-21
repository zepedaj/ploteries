from unittest import TestCase
import warnings
from ploteries2 import writer as mdl
from ploteries2.figure_managers import SmoothenedScalarsManager
from tempfile import NamedTemporaryFile
import sqlalchemy as sqa
import numpy as np
import numpy.testing as npt
from pglib.py import SliceSequence
from plotly import graph_objects as go
from ploteries2.reader import Reader


class TestWriter(TestCase):

    def test_open_existing(self):
        with NamedTemporaryFile() as tmpfo:
            #
            writer = mdl.Writer(tmpfo.name)
            writer.create_data_table('np_data1', np.ndarray)
            arr = np.array([0, 1, 2, 3])
            writer.add_data('np_data1', {'content': arr}, 0)
            writer.flush()
            writer.engine.dispose()
            #
            writer = mdl.Writer(tmpfo.name)
            table = writer.get_data_table('np_data1')
            npt.assert_equal(writer.execute(table.select())[0]['content'], arr)

    def test_create_data_table_and_add_data(self):
        with NamedTemporaryFile() as tmpfo:
            writer = mdl.Writer(tmpfo.name)
            writer.create_data_table('np_data1', np.ndarray)
            arr = np.array([0, 1, 2, 3])
            writer.add_data('np_data1', {'content': arr}, 0)
            writer.flush()

            with writer.engine.begin() as conn:
                out = conn.execute(sqa.select(
                    [writer.get_data_table('np_data1').c.content])).fetchall()
            npt.assert_equal(out[0][0], arr)

    def test_register_figure(self):
        with NamedTemporaryFile() as tmpfo:
            writer = mdl.Writer(tmpfo.name)
            writer.create_data_table('np_data1', np.ndarray)

            figure = go.Figure()
            #
            np_data1 = writer.get_data_table('np_data1')

            # Check 0
            out = writer.execute(sqa.select(
                [sqa.func.count(writer._figures).label('count')]))
            self.assertEqual(out[0]['count'], 0)
            out = writer.execute(
                sqa.select([sqa.func.count(writer._data_templates).label('count')]))
            self.assertEqual(out[0]['count'], 0)

            # Fail registration of new display, check  no inconsistent state
            try:
                writer.register_figure(
                    'plots/figure1', figure, SmoothenedScalarsManager,
                    [(None, [(['data', 0, 'x'], ['content'], 'error_source')])])
                raise Exception('Should raise error.')
            except sqa.exc.StatementError as err:
                if not isinstance(err.orig, ValueError) or err.orig.args != (
                        'too many values to unpack (expected 2)',):
                    raise
            out = writer.execute(
                sqa.select([sqa.func.count(writer._figures).label('count')]))
            self.assertEqual(out[0]['count'], 0)
            out = writer.execute(
                sqa.select([sqa.func.count(writer._data_templates).label('count')]))
            self.assertEqual(out[0]['count'], 0)

            # Successfully register new display.
            orig_sql = sqa.select([np_data1.c.content, np_data1.c.global_step])
            writer.register_figure(
                'plots/figure2', figure, SmoothenedScalarsManager,
                [(orig_sql, [(['data', 0, 'x'], ['content'])])])

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

    def test_add_methods_were_registered(self):
        #
        mdl.Writer.add_scalar
        mdl.Writer.add_scalars

        with NamedTemporaryFile() as tmpfo:
            writer = mdl.Writer(tmpfo.name)
            writer.add_scalars('scalars1', np.array([0, 1, 2]), 0)
            writer.add_scalars('scalars1', np.array([3, 4, 5]), 1)
            writer.flush()

            # Verify figure exists.
            dat = writer.execute(sqa.select(
                [writer._figures]).where(sqa.column('tag') == 'scalars1'))
            self.assertEqual(len(dat), 1)

    def test_rollback_table_creation(self):
        with NamedTemporaryFile() as tmpfo:
            writer = mdl.Writer(tmpfo.name)
            try:
                with writer.engine.begin() as conn:
                    writer.create_data_table(
                        'abc', np.ndarray, connection=conn)
                    raise Exception('Dummy')
            except Exception as err:
                if err.args[0] != 'Dummy':
                    raise

            with self.assertRaises(KeyError):
                writer.get_data_table('abc')

    def test_rollback_table_creation(self):
        if True:
            warnings.warn(
                'Skipping current test due to lack of table creation rollback support in sqlite3.')
            return
        with NamedTemporaryFile() as tmpfo:
            writer = mdl.Writer(tmpfo.name)
            try:
                with writer.engine.begin() as conn:
                    writer.create_data_table(
                        'abc', np.ndarray, connection=conn)
                    raise Exception('Dummy')
            except Exception as err:
                if err.args[0] != 'Dummy':
                    raise

            with self.assertRaises(KeyError):
                writer.get_data_table('abc')

    def test_rollback_table_creation_sqaalchemy(self):
        if True:
            warnings.warn(
                'Skipping current test due to lack of table creation rollback support in sqlite3.')
            return
        from sqlalchemy import inspect   # need to be running 0.8 for this

        with NamedTemporaryFile() as tmpfo:
            engine = sqa.create_engine('sqlite:///' + tmpfo.name)
            metadata = sqa.MetaData()
            table = sqa.Table('test_table', metadata,
                              sqa.Column('id', sqa.Integer, primary_key=True),
                              sqa.Column('tag', sqa.types.String,
                                         unique=True, nullable=False))

            with engine.connect() as conn:
                trans = conn.begin()
                metadata.create_all(conn)
                inspector = inspect(conn)
                table_names = inspector.get_table_names()
                trans.rollback()

                inspector = inspect(conn)
                rolled_back_table_names = inspector.get_table_names()
                print(rolled_back_table_names)

    def test_rollback_table_creation_sqaalchemy(self):
        if True:
            warnings.warn(
                'Skipping current test due to lack of table creation rollback support in sqlite3.')
            return
        from sqlalchemy import inspect   # need to be running 0.8 for this

        with NamedTemporaryFile() as tmpfo:
            engine = sqa.create_engine('sqlite:///' + tmpfo.name)
            metadata = sqa.MetaData()
            table = sqa.Table('test_table', metadata,
                              sqa.Column('id', sqa.Integer, primary_key=True),
                              sqa.Column('tag', sqa.types.String,
                                         unique=True, nullable=False))

            with engine.connect() as conn:
                trans = conn.begin()
                metadata.create_all(conn)
                inspector = inspect(conn)
                table_names = inspector.get_table_names()
                trans.rollback()

                inspector = inspect(conn)
                rolled_back_table_names = inspector.get_table_names()
                print(rolled_back_table_names)

    # def test_float_type(self):
    #     with NamedTemporaryFile() as tmpfo:
    #         writer = mdl.Writer(tmpfo.name)
    #         writer.add_scalars('float_scalars', 1.0, 0)
    #         self.assertEqual(
    #             writer._metadata.tables['float_scalars'].c.type, sqa.types.Float)
