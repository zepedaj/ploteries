from xerializer.abstract_type_serializer import TypeSerializer
from numpy.lib.format import descr_to_dtype
import base64


class LegacyTypeSerializer(TypeSerializer):

    def _build_obj(self, value, from_serializable):
        value = value['__value__']
        return self.from_serializable(from_serializable(value))


class SliceSerializer(LegacyTypeSerializer):

    handled_type = None
    as_serializable = None
    signature = 'pglib.serializer.extensions.SliceSerializer'

    def from_serializable(self, value):
        return slice(*value)


class SliceSequenceSerializer(LegacyTypeSerializer):

    handled_type = None
    as_serializable = None
    signature = 'pglib.slice_sequence.SliceSequence'

    def from_serializable(self, value):
        from ploteries.data_store.data_store import Ref_
        return Ref_.produce(value)


class PloteriesRefSerializer(SliceSequenceSerializer):
    signature = 'ploteries.data_store.data_store.Ref_'


class DtypeSerializer(LegacyTypeSerializer):

    handled_type = None
    as_serializable = None
    signature = 'pglib.serializer.extensions.DtypeSerializer'

    def from_serializable(self, value):
        return descr_to_dtype(value)


class NDArraySerializer(LegacyTypeSerializer):

    handled_type = None
    as_serializable = None
    signature = 'pglib.serializer.extensions.NDArraySerializer'

    def from_serializable(self, value):
        from pglib.numpy import decode_ndarray
        return decode_ndarray(base64.b64decode(value.encode('ascii')))


class TupleSerializer(LegacyTypeSerializer):
    handled_type = tuple
    as_serializable = None
    signature = 'builtins.tuple'

    def from_serializable(self, value):
        return self.handled_type(value)


class SetSerializer(TupleSerializer):
    handled_type = set
    signature = 'builtins.set'


class ListSerializer(TupleSerializer):
    handled_type = list
    signature = 'builtins.list'
