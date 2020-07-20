from pglib.nnets.torch.train.viz import Viz as _Viz, ScalarAccumViz as _ScalarAccumViz, default_viz_fxn
from pglib.nnets import numtor as nt


class GenericScalarsManagerAccumViz(_ScalarAccumViz):

    def write_epoch(self, writer, k_sample, name=None, **name_formatters):
        name = self.get_name(name, **name_formatters)
        writer.add_generic_scalars(
            name, self.get(), k_sample, **self.viz._extra_add_generic_kwargs)


class GenericScalarsManagerViz(_Viz):
    accum_cls = GenericScalarsManagerAccumViz

    def __init__(self, name, fxns, names=None, trace_kwargs=None):
        """
        Supports ploteries2.figure_managers.GenericScalarsManagerViz visualizations.
        """
        self.name, self.fxns = name, [default_viz_fxn(fxn) for fxn in fxns]
        self._extra_add_generic_kwargs = {
            'names': names, 'trace_kwargs': trace_kwargs}

    def update(self, batch, prediction, loss, timing, extra):
        # self.val=nt.copy(self.fxn(batch, prediction, loss, timing, extra))
        self.val = [
            nt.cpu(nt.detach(fxn(batch, prediction, loss, timing, extra)))
            for fxn in self.fxns]

    def get(self):
        return self.val

    def write_batch(self, writer, k_sample, name=None, **name_formatters):
        name = self.get_name(name, **name_formatters)
        # getattr(writer, self._add_method)(name, self.get(), k_sample)
        writer.add_generic_scalars(
            name, self.get(), k_sample, **self._extra_add_generic_kwargs)
