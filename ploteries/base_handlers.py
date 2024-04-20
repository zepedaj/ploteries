import abc
import jztools.validation as pgval
import re
from sqlalchemy.engine.result import Row
from sqlalchemy import insert, update, func, select, exc
from typing import Union
import numpy as np
from .data_store import Col_
from logging import getLogger

LOGGER = getLogger(__name__)


class Handler(abc.ABC):
    """
    Base class for :class:`~ploteries.figure_handler.FigureHandler`s and :class:`DataHandler`s
    """

    @classmethod
    @abc.abstractmethod
    def from_def_record(cls, data_store, data_def_record):
        """
        Initializes a data handler object from a raw (encoded) record from the data store's :attr:`data_defs` table.
        """

    @classmethod
    def from_name(cls, data_store, name, connection=None):
        """
        Loads the handler starting with the handler name.
        """
        with data_store.begin_connection(connection) as connection:
            def_record = connection.execute(
                select(cls.get_defs_table(data_store)).where(Col_("name") == name)
            ).one()
        return cls.from_def_record(data_store, def_record)

    @property
    @abc.abstractmethod
    def data_store(self):
        """
        Contains a :class:`~ploteries.data_store.DataStore` object.
        """

    @property
    @abc.abstractmethod
    def decoded_data_def(self):
        """
        Contains the Row retrieved from the data_defs table, with decoded parameters.
        """

    @classmethod
    @abc.abstractmethod
    def get_defs_table(cls, data_store):
        """
        Returns the defs table (e..g., data_defs or figure_defs)
        """

    @classmethod
    def decode_params(cls, params):
        """
        In-place decoding of the the params field of the data_defs record.
        """

    def encode_params(self, **extra_params):
        """
        Produces the params field to place in the data_defs record.
        """
        return None

    @classmethod
    def load_decode_def(
        cls, data_store, name, connection=None, check=True
    ) -> Union[bool, Row]:
        """
        Loads and decodes the definition from the the data defs table, if it exists, returning
        the decoded data def row if successful.

        Returns False if no data def is found in the table, or the data def row if it is found.
        """

        with data_store.begin_connection(connection) as conn:
            # Build query
            qry = (
                (defs_table := cls.get_defs_table(data_store))
                .select()
                .where(defs_table.c.name == name)
            )

            # Check that the store specs match the input.
            try:
                data_def = conn.execute(qry).one()
            except exc.NoResultFound:
                return False
            else:
                cls.decode_params(data_def._mapping["params"])
                return data_def

    def write_def(self, connection=None, mode="insert", extra_params={}):
        """
        Adds an entry to the data defs table and returns True if successful. If an entry already exists, returns False.
        """

        pgval.check_option("mode", mode, ["insert", "update"])

        record_dict = {
            "name": self.name,
            "handler": type(self),
            "params": self.encode_params(**extra_params),
        }

        with self.data_store.begin_connection(connection) as connection:
            defs_table = self.get_defs_table(self.data_store)
            if mode == "insert":
                try:
                    connection.execute(insert(defs_table), record_dict)
                except exc.IntegrityError as err:
                    if re.match(
                        f"\\(sqlite3.IntegrityError\\) UNIQUE constraint failed\\: {defs_table.name}\\.name",
                        str(err),
                    ):
                        return False
                    else:
                        raise

                else:
                    return True
            elif mode == "update":
                if self.decoded_data_def is None:
                    raise Exception(
                        "Cannot update a definition that has not been retrieved from the data store."
                    )
                connection.execute(
                    update(defs_table)
                    .where(defs_table.c.id == self.decoded_data_def.id)
                    .values(**record_dict)
                )
            else:
                raise Exception("Unexpected case.")


class DataHandler(Handler):
    @classmethod
    def get_defs_table(cls, data_store):
        return data_store.data_defs_table

    @abc.abstractmethod
    def encode_record_bytes(self, record_data) -> bytes:
        """
        Encodes the record's data to bytes to be added to the ``'bytes'`` field of the :attr:`data_records` table.
        """

    @abc.abstractmethod
    def decode_record_bytes(self, record_bytes: bytes):
        """
        Decodes the record's ``'bytes'`` field to produce the record's data.
        """

    def add_data(self, index, record_data, connection=None):
        """
        Add new data row.
        """

        # Encoding enables support for lazy initialization (e.g., upon calling method add in class UniformNDArrayDataHandler)
        encoded_data = self.encode_record_bytes(record_data)

        # Convert data, build records
        record = {
            "index": index,
            "writer_id": self.data_store.writer_id,
            "data_def_id": self.decoded_data_def.id,
            "bytes": encoded_data,
        }
        # records = [{'row_bytes': np.ascontiguousarray(recfns.repack_fields(arr_row)).tobytes()} for arr_row in arr]

        # Write to database.
        self.data_store.insert_data_record(record)
        # with self.data_store.begin_connection(connection) as connection:
        #    connection.execute(insert(self.data_store.data_records_table), record)

    def load_data(self, *criterion, single_record=False, connection=None):
        """
        By default, loads all the records owned by this handler.

        :param criterion: Passed as extra args to a where clause to further restrict the number of records
        """
        with self.data_store.begin_connection(connection) as connection:
            qry = select(self.data_store.data_records_table).where(
                self.data_store.data_records_table.c.data_def_id
                == self.decoded_data_def.id,
                *criterion,
            )
            if single_record:
                qry = qry.order_by(
                    self.data_store.data_records_table.c.index.desc()
                ).first()
            else:
                qry = qry.order_by(self.data_store.data_records_table.c.index.asc())

            records = list(connection.execute(qry))

        return self._format_records(records)

    def _format_records(self, records):
        meta = np.empty(
            len(records),
            dtype=[("index", "i8"), ("created", "datetime64[us]"), ("writer_id", "i")],
        )
        meta["index"] = [_rec.index for _rec in records]
        meta["created"] = [_rec.created for _rec in records]
        meta["writer_id"] = [_rec.writer_id for _rec in records]

        return {
            "meta": meta,
            "data": self.merge_records_data(
                [self.decode_record_bytes(_rec.bytes) for _rec in records]
            ),
        }

    @abc.abstractmethod
    def merge_records_data(self, records_data):
        """
        Merges the list of decoded record bytes to create the ``'data'`` field of the dictionary output by the load function.
        """

    def __len__(self, connection=None):
        """
        Returns the number of records in the array (i.e., the first dimension of the array). The shape of the entire stored array is hence (len(self), *self.recdims).
        """
        with self.lock:
            if self.decoded_data_def is None:
                return 0
            else:
                with self.data_store.begin_connection(connection) as connection:
                    return connection.execute(
                        select(
                            func.count(self.data_store.data_records_table.c.id)
                        ).where(
                            self.data_store.data_records_table.c.data_def_id
                            == self.decoded_data_def["id"]
                        )
                    ).scalar()
