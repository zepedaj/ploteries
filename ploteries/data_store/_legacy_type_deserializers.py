from xerializer.abstract_type_serializer import TypeSerializer
from xerializer import default_signature, create_signature_aliases
from numpy.lib.format import descr_to_dtype
import base64
from importlib import import_module

for legacy_signature in ["ploteries.ndarray_data_handlers.NDArraySpec"]:
    module_name, class_name = legacy_signature.rsplit(".", 1)
    module = import_module(module_name)
    create_signature_aliases(
        default_signature(getattr(module, class_name)), legacy_signature
    )


class LegacyTypeSerializer(TypeSerializer):
    def _build_obj(self, value, from_serializable):
        value = value["__value__"]
        return self.from_serializable(from_serializable(value))


class SliceSerializer(LegacyTypeSerializer):

    handled_type = None
    as_serializable = None
    signature = "jztools.serializer.extensions.SliceSerializer"

    def from_serializable(self, value):
        return slice(*value)


class SliceSequenceSerializer(LegacyTypeSerializer):

    handled_type = None
    as_serializable = None
    signature = "jztools.slice_sequence.SliceSequence"

    def from_serializable(self, value):
        from ploteries.data_store.data_store import Ref_

        return Ref_.produce(value)


class PloteriesRefSerializer(SliceSequenceSerializer):
    signature = "ploteries.data_store.data_store.Ref_"


class DtypeSerializer(LegacyTypeSerializer):

    handled_type = None
    as_serializable = None
    signature = "jztools.serializer.extensions.DtypeSerializer"

    def from_serializable(self, value):
        return descr_to_dtype(value)


class NDArraySerializer(LegacyTypeSerializer):

    handled_type = None
    as_serializable = None
    signature = "jztools.serializer.extensions.NDArraySerializer"

    def from_serializable(self, value):
        from jztools.numpy import decode_ndarray

        return decode_ndarray(base64.b64decode(value.encode("ascii")))


class TupleSerializer(LegacyTypeSerializer):
    handled_type = tuple
    as_serializable = None
    signature = "builtins.tuple"

    def from_serializable(self, value):
        return self.handled_type(value)


class SetSerializer(TupleSerializer):
    handled_type = set
    signature = "builtins.set"


class ListSerializer(TupleSerializer):
    handled_type = list
    signature = "builtins.list"
