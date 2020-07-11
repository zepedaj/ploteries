from sqlalchemy.types import TypeDecorator, VARCHAR
from sqlalchemy.ext import serializer as sqla_serializer
from pglib.py import SliceSequence
import json


class DataMapperType(TypeDecorator):
    """
    Represents multiple mappings from an sql query to fields in a plotly figure.data structure
    as a list of 2-tuple of SliceSequence objects.
    """

    impl = VARCHAR

    def process_bind_param(self, value, dialect):

        value = [(SliceSequence(figure_seq).serialize(), SliceSequence(sql_query_seq).serialize())
                 for figure_seq, sql_query_seq in value]
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        value = json.loads(value)
        value = [(SliceSequence.deserialize(figure_seq), SliceSequence.deserialize(sql_query_seq))
                 for figure_seq, sql_query_seq in value]

        return value


def sql_query_type_builder(session_class, bound_metadata):
    class SQLQueryType(TypeDecorator):
        impl = VARCHAR
        Session = session_class
        metadata = bound_metadata

        def process_bind_param(self, value, dialect):
            if isinstance(value, tuple):
                if len(value) != 1:
                    raise Exception('Invalid input')
                value = value[0]
            return sqla_serializer.dumps(value)

        def process_result_value(self, value, dialect):
            return sqla_serializer.loads(value, self.metadata, self.Session)
    return SQLQueryType
