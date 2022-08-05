from .ndarray_data_handlers import RaggedNDArrayDataHandler as _RaggedNDArrayDataHandler
from xerializer import Serializer as _Serializer


class SerializableDataHandler(_RaggedNDArrayDataHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._serializer = _Serializer()

    def encode_record_bytes(self, val):
        return self._serializer.serialize(val).encode("utf-8")

    def decode_record_bytes(self, val):
        return self._serializer.deserialize(val.decode("utf-8"))
