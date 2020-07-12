from sqlalchemy.types import TypeDecorator, VARCHAR
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
