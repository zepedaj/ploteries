from unittest import TestCase
from ploteries2.writer import Writer
from ploteries2 import reader as mdl
from ploteries2.figure_managers import SmoothenedScalarsManager
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
            figure_recs = writer.load_figure_recs(
                manager=SmoothenedScalarsManager)
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
            figure_recs = reader.load_figure_recs(
                manager=SmoothenedScalarsManager)
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

    def test_outer_join_data_tables__equals_inner(self):
        self.test_join_data_tables(join_fxn='outer_join_data_tables')

    def test_join_data_tables(self, join_fxn='join_data_tables'):
        with NamedTemporaryFile() as tmpfo:
            # Create tables
            writer = Writer(tmpfo.name)

            def gen_data(N): return [np.random.randn(np.random.randint(
                2, 10), np.random.randint(2, 10)) for k in range(N)]
            content0 = gen_data(2)
            writer.add_data('table0', {'content': content0[0]}, 0)
            writer.add_data('table0', {'content': content0[1]}, 1)

            content1 = gen_data(2)
            writer.add_data('table1', {'content': content1[0]}, 0)
            writer.add_data('table1', {'content': content1[1]}, 1)

            # Outer joins don't work in sqlalchemy...
            # content2 = gen_data(2)
            # writer.add_data('table2', content2[0], 0)
            # # Only content2 has global_step=2
            # writer.add_data('table2', content2[1], 2)

            writer.flush()

            # Create outer join from reader
            reader = mdl.Reader(tmpfo.name)
            sql = getattr(reader, join_fxn)([
                reader.get_data_table('table0'),
                reader.get_data_table('table1')])

            data = reader.execute(sql)
            self.assertEqual(len(data), 2)

            # Global step 0
            self.assertEqual(set(data[0].keys()), {
                             'global_step', 'content0', 'content1'})
            npt.assert_equal(data[0]['content0'], content0[0])
            npt.assert_equal(data[1]['content0'], content0[1])

            # Global step 1
            self.assertEqual(set(data[1].keys()), {
                             'global_step', 'content0', 'content1'})
            npt.assert_equal(data[0]['content1'], content1[0])
            npt.assert_equal(data[1]['content1'], content1[1])

    def test_all_join_types(self):
        rng = np.random.RandomState(0)
        with NamedTemporaryFile() as tmpfo:
            # Create tables
            writer = Writer(tmpfo.name)

            def gen_data(N): return [rng.randn(rng.randint(
                2, 10), rng.randint(2, 10)) for k in range(N)]
            content0 = gen_data(2)
            writer.add_data('table0', {'content': content0[0]}, 0)
            writer.add_data('table0', {'content': content0[1]}, 1)

            content1 = gen_data(2)
            writer.add_data('table1', {'content': content1[0]}, 0)
            writer.add_data('table1', {'content': content1[1]}, 1)

            content2 = gen_data(2)
            writer.add_data('table2', {'content': content2[0]}, 0)
            # Only content2 has global_step=2
            writer.add_data('table2', {'content': content2[1]}, 2)

            writer.flush()

            # Test left outer join
            reader = mdl.Reader(tmpfo.name)
            sql = reader.join_data_tables([
                (reader.get_data_table('table2'), 'content2'),
                (reader.get_data_table('table0'), 'content0'),
                (reader.get_data_table('table1'), 'content1')
            ], isouter=True)
            data = reader.execute(sql.order_by(sql.c.global_step))
            self.assertEqual([_d['global_step'] for _d in data], [0, 2])
            self.assertEqual(
                [_d['content0'] is None for _d in data], [False, True])

            # Test inner join
            sql = reader.join_data_tables([
                (reader.get_data_table('table1'), 'content1'),
                (reader.get_data_table('table2'), 'content2'),
                (reader.get_data_table('table0'), 'content0')
            ], isouter=False)
            data = reader.execute(sql.order_by(sql.c.global_step))
            self.assertEqual([_d['global_step'] for _d in data], [0])
            self.assertEqual(
                [_d['content0'] is None for _d in data], [False])
            self.assertEqual(
                [_d['content2'] is None for _d in data], [False])

            # Test full outer join
            sql = reader.outer_join_data_tables([
                (reader.get_data_table('table1'), 'content1'),
                (reader.get_data_table('table2'), 'content2'),
                (reader.get_data_table('table0'), 'content0')
            ])
            data = reader.execute(sql.order_by(sql.c.global_step))

            self.assertEqual([_d['global_step'] for _d in data], [0, 1, 2])
            self.assertEqual(
                [_d['content0'] is None for _d in data], [False, False, True])
            self.assertEqual(
                [_d['content1'] is None for _d in data], [False, False, True])
            self.assertEqual(
                [_d['content2'] is None for _d in data], [False, True, False])
            npt.assert_equal(
                [_d['content0'] for _d in data if _d['content0'] is not None], content0)
            npt.assert_equal(
                [_d['content1'] for _d in data if _d['content1'] is not None], content1)
            npt.assert_equal(
                [_d['content2'] for _d in data if _d['content2'] is not None], content2)
