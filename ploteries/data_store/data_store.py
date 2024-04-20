from sqlalchemy import (
    func,
    Table,
    Column,
    Integer,
    String,
    DateTime,
    select,
    ForeignKey,
    LargeBinary,
    create_engine,
    MetaData,
    and_,
    UniqueConstraint,
)
import numpy as np
import itertools as it
from jztools.validation import checked_get_single
from jztools.sqlalchemy import ClassType, create_serializable_type, ThreadedInsertCache
from contextlib import contextmanager
from jztools.sqlalchemy import begin_connection
from sqlalchemy.sql.elements import BinaryExpression
from sqlalchemy.sql import column
from typing import Union, List, Tuple
from jztools.slice_sequence import SSQ_ as _SSQ_
from xerializer.abstract_type_serializer import Serializable

Col_ = column
"""
Convenience alias to :class:`sqlalchemy.sql.column`. See documentation for :meth:`DataStore.get_data_handlers` for example usage.
"""


class Ref_(_SSQ_, Serializable):
    """
    Extension to :class:`~jztools.slice_sequence.SliceSequence` that is used to define serializable references to data store content.
    """

    def __init__(self, index):
        """
        :params data, index: Same params taken by dictionary form input to :meth:`DataStore.__getitem__`.
        """
        super().__init__()
        index, multi_series = DataStore.format_getitem_idx(index)
        if not multi_series:
            index["data"] = index["data"][0]
        self.slice_sequence.append(index)

    @property
    def query_params(self):
        return self.slice_sequence[0]

    @classmethod
    def call_multi(cls, data_store: "DataStore", *refs_: "Ref_", _test_output=False):
        """
        Applies the multiple Ref_ slice sequences to the data store individually, but avoids redundant queries that are repeated across refs_.
        """

        # Get the source data without repeating queries.
        source_data_pairs = []
        num_retrievals = 0
        for _ref in refs_:
            new_source = Ref_.produce(_ref.slice_sequence[:1])
            try:
                data = next(filter(lambda _x: _x[0] == new_source, source_data_pairs))[
                    1
                ]
            except StopIteration:
                num_retrievals += 1
                data = new_source(data_store)
            source_data_pairs.append((new_source, data))

        # Apply remainder of slice sequence.
        remainders = [Ref_.produce(_ref.slice_sequence[1:]) for _ref in refs_]

        output = [
            _remainder(_source_data)
            for (_source_ref, _source_data), _remainder in zip(
                source_data_pairs, remainders
            )
        ]

        if _test_output:
            return source_data_pairs, num_retrievals, remainders, output
        else:
            return output


class DataStore:
    """
    Stores figure definitions, data series definitions and data series records. Each of these is stored in one of three tables. The data series records table in particular contains heterogenoeous data records for all data series types. Exposes various facilities including

    * an :meth:`insert_data_record` method that supports asynchronous data record insert and
    * a :meth:`__getitem__` method that makes it possible to retrieve all records or a subset thereof from one or more data series that are joined on the time step using familiar bracket syntax (e.g., ``data_store['data_series_0']`` ).
    * The :meth:`__getitem__` method also makes it possible to employ :class:`Ref_` objects embedded in figure definitions that retrieve data from the data store when building the figure.

    .. todo:: Add an example below.

    .. ipython::

        In [304]: print('Hello world')
        Hello world
    """

    def __init__(self, path, read_only=False, max_queue_size=1000):
        """
        :param path: Database path.
        :param read_only: Use database in read-only mode.
        :param max_queue_size: Max size of insert cache queue.
        """
        #
        if read_only:
            with open(path, "r"):
                pass
            self.writer_id = None
            self.threaded_insert_cache = None
        #
        self.path = path
        self.engine = create_engine(f"sqlite:///{path}")
        self._metadata = MetaData()

        #
        self._metadata.reflect(self.engine)
        self._create_tables()

        # Set writer instance
        if not read_only:
            with self.engine.connect() as conn, conn.begin():
                self.writer_id = conn.execute(
                    self._metadata.tables["writers"].insert()
                ).inserted_primary_key.id
            #
            self.threaded_insert_cache = ThreadedInsertCache(
                self.data_records_table, self.engine, max_queue_size
            )

    @contextmanager
    def begin_connection(self, connection=None):
        with begin_connection(self.engine, connection) as connection:
            yield connection

    def _get_handlers(
        self, defs_table, *column_constraints: BinaryExpression, connection=None
    ):
        """
        Helper for :meth:`get_data_handlers` and :meth:`get_figure_handlers`
        """
        with self.begin_connection(connection) as connection:
            handlers = list(
                (
                    _rec.handler.from_def_record(self, _rec)
                    for _rec in connection.execute(
                        select(defs_table).where(*column_constraints)
                    )
                )
            )
        return handlers

    def get_data_handlers(self, *column_constraints: BinaryExpression, connection=None):
        """
        Gets the data handlers satisfying the specified binary constraints. E.g.,

        ```
        from ploteries.data_store import col
        ```

        * ``get_data_handlers()`` returns all handlers,
        * ``get_data_handlers(Col_('name')=='arr1')`` returns the data handler of name 'arr1',
        * ``get_data_handlers(data_store.data_defs_table.c.name=='arr1')`` returns the data handler of name 'arr1',
        * ``get_data_handlers(Col_('type')==UniformNDArrayDataHandler)`` returns all data handlers of that type. (NOT WORKING!)

        .. todo:: Type constraints are not working (see last bullet above).

        """
        return self._get_handlers(
            self.data_defs_table, *column_constraints, connection=connection
        )

    def get_figure_handlers(
        self, *column_constraints: BinaryExpression, connection=None
    ):
        """
        Gets the figure handlers satisfying the specified binary constraints. See :meth:`get_data_handlers` for an example.
        """

        return self._get_handlers(
            self.figure_defs_table, *column_constraints, connection=connection
        )

    def insert_data_record(self, data_record: Union[dict, List[dict]]):
        """
        Using this method will automatically batch inserts in a new thread to increase disk-write efficiency (:class:`~jztools.sqlalchemy.threaded_insert_cache.ThreadedInsertCache` used internally). Call :meth:`flush` to ensure all records inserted before the call have been written to the database.

        :param data_record: Record dictionary or list of record dictionaries.
        """
        if self.threaded_insert_cache is None:
            raise Exception(
                "Attempting to insert into a data store that was opened in read-only mode."
            )
        else:
            self.threaded_insert_cache.insert(data_record)

    def flush(self):
        if self.threaded_insert_cache:
            self.threaded_insert_cache.flush()

    def _create_tables(self):
        """
        Creates new tables or sets their column type.
        """

        self.data_records_table = Table(
            "data_records",
            self._metadata,
            Column("id", Integer, primary_key=True),
            Column("index", Integer, nullable=False),
            Column("created", DateTime, server_default=func.now(), nullable=False),
            Column("writer_id", ForeignKey("writers.id"), nullable=False),
            Column("data_def_id", ForeignKey("data_defs.id"), nullable=False),
            Column("bytes", LargeBinary),
            UniqueConstraint(
                "index", "writer_id", "data_def_id", name="uix_index_writer_data_def"
            ),
            extend_existing=True,
        )

        # Distinguishes between writing form different DataStore instances.
        self.writers_table = Table(
            "writers",
            self._metadata,
            Column("id", Integer, primary_key=True),
            Column("created", DateTime, server_default=func.now(), nullable=False),
            extend_existing=True,
        )

        # Specifies how to retrieve and decode data bytes from the data_records table
        self.data_defs_table = Table(
            "data_defs",
            self._metadata,
            Column("id", Integer, primary_key=True),
            Column("name", String, unique=True),
            Column("handler", ClassType, nullable=False),
            Column("params", create_serializable_type(), nullable=True),
            extend_existing=True,
        )

        # Specifies figure creation from stored data.
        self.figure_defs_table = Table(
            "figure_defs",
            self._metadata,
            Column("id", Integer, primary_key=True),
            Column("name", String, unique=True),
            Column("handler", ClassType, nullable=False),
            Column("params", create_serializable_type()),
            extend_existing=True,
        )

        self._metadata.create_all(self.engine)

    @classmethod
    def format_getitem_idx(cls, idx):
        multi_series = True
        if isinstance(idx, str):
            multi_series = False
            idx = {"data": [idx]}
        elif isinstance(idx, (list, tuple)):
            idx = {"data": idx}
        elif isinstance(idx, dict) and isinstance(idx["data"], str):
            idx = dict(idx)
            multi_series = False
            idx["data"] = [idx["data"]]
        elif not isinstance(idx, dict):
            raise TypeError(f"Invalid input type {type(idx)}.")
        idx = {"connection": None, "criterion": [], "index": None, **idx}
        if idx["index"] not in ("latest", None) and not isinstance(idx["index"], int):
            raise ValueError(f"Invalid value for index = {idx['index']}.")

        # Remove redundant data names and sort.
        # Sorting helps call_multi avoid repeated queries.
        idx["data"] = sorted(set(idx["data"]))

        return idx, multi_series

    def __getitem__(self, idx: Union[str, Tuple[str], dict]):
        """
        Load the data in a table or table join. This method can load all the data or a single record. Joins are carried out on data records table fields :attr:`index` and :attr:`writer_id`.

        The returned output is a dictionary in one of two formats: When data series names are provided as a tuple of strings, the format is

        .. code-block::

            {'meta': numpy.ndarray(shape=num_records,
                                   dtype=[('index', '<i4'), ('writer_id', '<i4')]),
             'series': {
                 'series_name_1': {
                     'created': numpy.ndarray(shape=num_records,
                                              dtype='datetime64[us]'),
                     'data': (data handler dependent content of length num_records)},
                 'series_name_2': ...}
             }

        When a single data series name is provided as a string, the nested :attr:`data` and :attr:`created` fields are provided at the root level:

        .. code-block::

            {'meta': numpy.ndarray(shape=num_records,
                                   dtype=[('index', '<i4'), ('writer_id', '<i4')]),
             'created': numpy.ndarray(shape=num_records, dtype='datetime64[us]'),
             'data': (data handler dependent content of length num_records)
             }

        :param idx: Data name or tuple of data names (to specify a join). Alternatively, pass the data name (tuple) as field 'data' in a dictionary that can further contain field 'criterion' to specify any further criterion.

        * 'data': Data name string or list of names.
        * 'index': If set to 'latest', will return a record with the highest index (there might be more than one, of which one of those with the highest worker_id is taken). Otherwise, needs to be an index value. Ignored if ``None`` (the default).
        * 'connection': Connection object from previously-started context if any. ``None`` by default.
        * 'criterion': List of extra criterion to apply to the data_records_table query. Empty list ``[]`` by default.
        """

        # Add default values to idx.
        idx, multi_series = self.format_getitem_idx(idx)

        # Build aliases and load data handlers.
        data_records_aliases = [
            {
                "name": name,
                "handler": (
                    handler := checked_get_single(
                        self.get_data_handlers(Col_("name") == name)
                    )
                ),
                "alias": select(self.data_records_table)
                .where(
                    self.data_records_table.c.data_def_id == handler.decoded_data_def.id
                )
                .alias(name),
            }
            for name in idx["data"]
        ]

        # Build query
        _last_table = data_records_aliases[0]["alias"]
        joined_tables = _last_table
        for _table_to_join in (_x["alias"] for _x in data_records_aliases[1:]):
            joined_tables = joined_tables.join(
                _table_to_join,
                (
                    and_(
                        _last_table.c.index == _table_to_join.c.index,
                        _last_table.c.writer_id == _table_to_join.c.writer_id,
                    )
                ),
            )

        qry = select(
            # Meta
            *[
                getattr(_last_table.c, col).label(f"{col}")
                for col in ["index", "writer_id"]
            ],
            # Series
            *it.chain(
                *[
                    [
                        getattr(_al["alias"].c, col).label(f"{_al['name']}.{col}")
                        for col in ["created", "bytes"]
                    ]
                    for _al in data_records_aliases
                ]
            ),
        ).select_from(joined_tables)

        #
        if idx["index"] is not None:
            # Get a single record.
            if idx["index"] == "latest":
                # Get the record with most recent index.
                qry = qry.order_by(Col_("index").desc()).limit(1)
            else:
                # Get the record with the provided index.
                qry = qry.where(_last_table.c.index == idx["index"])
        else:
            # Get all records in sorted order.
            qry = qry.order_by(Col_("index").asc(), Col_("writer_id").asc())

        # Execute query.
        with self.begin_connection(connection=idx["connection"]) as connection:
            records = connection.execute(qry).fetchall()

        # Format record meta data.
        meta = np.empty(len(records), dtype=[("index", "i8"), ("writer_id", "i8")])
        meta["index"] = [_rec.index for _rec in records]
        meta["writer_id"] = [_rec.writer_id for _rec in records]

        # Format record created time and bytes.
        series = {}
        for _al in data_records_aliases:
            name = _al["name"]
            #
            _content = {"created": np.empty(len(records), dtype="datetime64[us]")}
            _content["created"][:] = [
                _rec._mapping[f"{name}.created"] for _rec in records
            ]
            #
            _content["data"] = _al["handler"].merge_records_data(
                [
                    _al["handler"].decode_record_bytes(_rec._mapping[f"{name}.bytes"])
                    for _rec in records
                ]
            )
            #
            series[name] = _content

        if multi_series:
            # {'meta': meta, 'series': {<series name>: <series data>, ...}
            out = {"meta": meta, "series": series}
        else:
            # {'meta': meta, 'data': <series data>, 'created': <created time>}
            out = {
                "meta": meta,
                **series[idx["data"][0]],  # Adds fields 'created', 'data'
            }

        return out
