from sqlalchemy import (func, Table, Column, Integer, String, DateTime, select,
                        ForeignKey, LargeBinary, create_engine, MetaData, and_,
                        UniqueConstraint)
import numpy as np
import itertools as it
from pglib.validation import checked_get_single
from pglib.sqlalchemy import ClassType, SerializableType
from pglib.serializer import Serializer as _Serializer
from contextlib import contextmanager
from pglib.sqlalchemy import begin_connection
from sqlalchemy.sql.elements import BinaryExpression
from sqlalchemy.sql import column
from typing import Union
from pglib.slice_sequence import SSQ_ as _SSQ_

Col_ = column
"""
Convenience alias to :class:`sqlalchemy.sql.column`. See documentation for :meth:`DataStore.get_data_handlers` for example usage.
"""


class Ref_(_SSQ_):
    """
    Extension to :class:`~pglib.slice_sequence.SliceSequence` that is used to define serializable references to data store content.
    """

    def __init__(self, data, index=None):
        """
        :params data, index: Same params taken by dictionary form input to :meth:`DataStore.__getitem__`.
        """
        super().__init__()
        idx, multi_series = DataStore.format_getitem_idx({'data': data, 'index': index})
        if not multi_series:
            idx['data'] = idx['data'][0]
        self.slice_sequence.append(idx)

    @property
    def query_params(self):
        return self.slice_sequence[0]

    @classmethod
    def call_multi(cls, data_store: 'DataStore', *refs_: 'Ref_', _output_all=False):
        """
        Applies the multiple Ref_ slice sequences to the data store individually, but avoids redundant queries that are repeated across refs_.
        """
        sources = [
            Ref_.produce(_ref.slice_sequence[:1])
            for _ref in refs_]
        remainders = [
            Ref_.produce(_ref.slice_sequence[1:])
            for _ref in refs_]
        source_to_data = {_source: _source(data_store) for _source in set(sources)}

        output = [_remainder(source_to_data[_source])
                  for _source, _remainder in zip(sources, remainders)]

        if _output_all:
            return {'sources': sources,
                    'remainders': remainders,
                    'source_to_data': source_to_data,
                    'output': output}
        else:
            return output


_Serializer.default_extension_types.append(Ref_)


class DataStore:
    def __init__(self, path, read_only=False):
        #
        if read_only:
            with open(path, 'r'):
                pass
            self.writer_id = None
        #
        self.path = path
        self.engine = create_engine(f'sqlite:///{path}')
        self._metadata = MetaData(bind=self.engine)

        #
        self._metadata.reflect()
        self._create_tables()

        # Set writer instance
        if not read_only:
            with self.engine.connect() as conn:
                self.writer_id = conn.execute(
                    self._metadata.tables['writers'].insert()).inserted_primary_key.id

    @contextmanager
    def begin_connection(self, connection=None):
        with begin_connection(self.engine, connection) as connection:
            yield connection

    def _get_handlers(self, defs_table, *column_constraints: BinaryExpression, connection=None):
        """
        Helper for :meth:`get_data_handlers` and :meth:`get_figure_handlers`
        """
        with self.begin_connection(connection) as connection:
            handlers = list((
                _rec.handler.from_def_record(self, _rec) for _rec in
                connection.execute(select(defs_table).where(*column_constraints))))
        return handlers

    def get_data_handlers(self, *column_constraints: BinaryExpression, connection=None):
        """
        Gets the data handlers satisfying the specified binary constraints. E.g.,

        ```
        from ploteries3.data_store import col
        ```

        * ``get_data_handlers()`` returns all handlers,
        * ``get_data_handlers(Col_('name')=='arr1')`` returns the data handler of name 'arr1',
        * ``get_data_handlers(data_store.data_defs_table.c.name=='arr1')`` returns the data handler of name 'arr1',
        * ``get_data_handlers(Col_('type')==UniformNDArrayDataHandler)`` returns all data handlers of that type. (NOT WORKING!)

        """
        return self._get_handlers(
            self.data_defs_table, *column_constraints, connection=connection)

    def get_figure_handlers(self, *column_constraints: BinaryExpression, connection=None):
        """
        Gets the figure handlers satisfying the specified binary constraints. See :method:`get_data_handlers` for an example.
        """

        return self._get_handlers(
            self.figure_defs_table, *column_constraints, connection=connection)

    def _create_tables(self):
        """
        Creates new tables or sets their column type.
        """

        self.data_records_table = Table(
            'data_records', self._metadata,
            Column('id', Integer, primary_key=True),
            Column('index', Integer, nullable=False),
            Column('created', DateTime, server_default=func.now(), nullable=False),
            Column('writer_id', ForeignKey('writers.id'), nullable=False),
            Column('data_def_id', ForeignKey('data_defs.id'), nullable=False),
            Column('bytes', LargeBinary),
            UniqueConstraint('index', 'writer_id', 'data_def_id', name='uix_index_writer_data_def'),
            extend_existing=True)

        # Distinguishes between writing form different DataStore instances.
        self.writers_table = Table(
            'writers', self._metadata,
            Column('id', Integer, primary_key=True),
            Column('created', DateTime, server_default=func.now(), nullable=False),
            extend_existing=True)

        # Specifies how to retrieve and decode data bytes from the data_records table
        self.data_defs_table = Table(
            'data_defs', self._metadata,
            Column('id', Integer, primary_key=True),
            Column('name', String, unique=True),
            Column('handler', ClassType, nullable=False),
            Column('params', SerializableType, nullable=True),
            extend_existing=True)

        # Specifies figure creation from stored data.
        self.figure_defs_table = Table(
            'figure_defs', self._metadata,
            Column('id', Integer, primary_key=True),
            Column('name', String, unique=True),
            Column('handler', ClassType, nullable=False),
            Column('params', SerializableType),
            extend_existing=True)

        self._metadata.create_all()

    @classmethod
    def format_getitem_idx(cls, idx):
        multi_series = True
        if isinstance(idx, str):
            multi_series = False
            idx = {'data': [idx]}
        elif isinstance(idx, (list, tuple)):
            idx = {'data': idx}
        elif (isinstance(idx, dict) and isinstance(idx['data'], str)):
            idx = dict(idx)
            multi_series = False
            idx['data'] = [idx['data']]
        elif not isinstance(idx, dict):
            raise TypeError(f'Invalid input type {type(idx)}.')
        idx = {
            'connection': None,
            'criterion': [],
            'index': None,
            **idx}
        if idx['index'] not in ('latest', None) and not isinstance(idx['index'], int):
            raise ValueError(f"Invalid value for index = {idx['index']}.")

        # Remove redundant data names and sort.
        # Sorting helps call_multi avoid repeated queries.
        idx['data'] = sorted(set(idx['data']))

        return idx, multi_series

    def __getitem__(self, idx: Union[str, tuple, dict]):
        """
        Load the data in a table or table join. Can load all the data or a single record. Joins are carried out on fields index and writer_id. Returns a dictionary in one of two formats. When data series names are provided as a tuple of strings, the format is:
        ```
        {'meta': <ndarray, shape=(num_records,), dtype=[('index', '<i4'), ('writer_id', '<i4')]>,
         'series': {
            <series name> : {
                     'created': <ndarray, shape=(num_records,), dtype='datetime64[us]'>,
                     'data': <data handler dependent content>}}}
        ```
        When a single data series name is provided as a string, the the nested 'data' and 'created' fields are provided at the root level:
        ```
        {'meta': <ndarray, shape=(num_records,), dtype=[('index', '<i4'), ('writer_id', '<i4')]>,
         'created': <ndarray, shape=(num_records,), dtype='datetime64[us]'>,
         'data': <data handler dependent content>}
        ```

        :param idx: Data name or tuple of data names (to specify a join). Alternatively, pass the data name (tuple) as field 'data' in a dictionary that can further contain field 'criterion' to specify any further criterion.

        * 'data': Data name string or list of names.
        * 'index': If set to 'latest', will return a record with the highest index (there might be more than one, of which one of those with the highest worker_id is taken). Otherwise, needs to be an index value. Ignored if ``None`` (the default).
        * 'connection': Connection object from previously-started context if any. ``None`` by default.
        * 'criterion': List of extra criterion to apply to the data_records_table query. Empty list ``[]`` by default.
       """

        # Add default values to idx.
        idx, multi_series = self.format_getitem_idx(idx)

        # Build aliases and load data handlers.
        data_records_aliases = [{
            'name': name,
            'handler': (handler := checked_get_single(
                self.get_data_handlers(Col_('name') == name))),
            'alias': select(self.data_records_table).where(
                self.data_records_table.c.data_def_id == handler.decoded_data_def.id).alias(name)}
            for name in idx['data']]

        # Build query
        _last_table = data_records_aliases[0]['alias']
        joined_tables = _last_table
        for _table_to_join in (_x['alias'] for _x in data_records_aliases[1:]):
            joined_tables = joined_tables.join(
                _table_to_join, (and_(_last_table.c.index == _table_to_join.c.index,
                                      _last_table.c.writer_id == _table_to_join.c.writer_id)))

        qry = select(
            # Meta
            *[getattr(_last_table.c, col).label(f'{col}')
              for col in ['index', 'writer_id']],
            # Series
            *it.chain(*[
                [getattr(_al['alias'].c, col).label(f"{_al['name']}.{col}")
                 for col in ['created', 'bytes']]
                for _al in data_records_aliases])).select_from(joined_tables)

        #
        if idx['index'] is not None:
            # Get a single record.
            if idx['index'] == 'latest':
                # Get the record with most recent index.
                qry = qry.order_by(Col_('index').desc()).limit(1)
            else:
                # Get the record with the provided index.
                qry = qry.where(_last_table.c.index == idx['index'])
        else:
            # Get all records in sorted order.
            qry = qry.order_by(Col_('index').asc(),
                               Col_('writer_id').asc())

        # Execute query.
        with self.begin_connection(connection=idx['connection']) as connection:
            records = connection.execute(qry).fetchall()

        # Format record meta data.
        meta = np.empty(len(records),
                        dtype=[('index', 'i'),
                               ('writer_id', 'i')])
        meta['index'] = [_rec.index for _rec in records]
        meta['writer_id'] = [_rec.writer_id for _rec in records]

        # Format record created time and bytes.
        series = {}
        for _al in data_records_aliases:
            name = _al['name']
            #
            _content = {'created': np.empty(len(records), dtype='datetime64[us]')}
            _content['created'][:] = [
                _rec[f'{name}.created'] for _rec in records]
            #
            _content['data'] = _al['handler'].merge_records_data([
                _al['handler'].decode_record_bytes(_rec[f'{name}.bytes']) for _rec in records])
            #
            series[name] = _content

        if multi_series:
            # {'meta': meta, 'series': {<series name>: <series data>, ...}
            out = {
                'meta': meta,
                'series': series
            }
        else:
            # {'meta': meta, 'data': <series data>, 'created': <created time>}
            out = {
                'meta': meta,
                **series[idx['data'][0]]  # Adds fields 'created', 'data'
            }

        return out
