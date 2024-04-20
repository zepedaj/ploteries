from torch_train_manager.viz import (
    Viz as _Viz,
    ScalarViz as _ScalarViz,
    ScalarAccumViz as _ScalarAccumViz,
    AccumViz as _AccumViz,
    default_viz_fxn,
)
from jztools.nnets import numtor as nt
import numpy as np
from .writer import Writer
import torch


class GenericScalarsManagerAccumViz(_ScalarAccumViz):
    def __init__(self, *args, **kwargs):
        self.smoothing = kwargs.pop("smoothing", True)
        super().__init__(*args, **kwargs)

    def write_epoch(self, writer, k_sample, name=None, **name_formatters):
        name = self.get_name(name, **name_formatters)
        writer.add_scalars(
            name,
            self.get(),
            k_sample,
            traces_kwargs=self.viz.traces_kwargs,
            smoothing=self.smoothing,
        )

    def get(self):
        if self.val is None and (
            traces_kwargs := getattr(self.viz, "traces_kwargs", None)
        ):
            return [np.nan] * len(traces_kwargs)
        else:
            return super().get()


class GenericScalarsManagerViz(_Viz):
    def accum_cls(self, *args, **kwargs):
        return GenericScalarsManagerAccumViz(
            *args, **{**{"smoothing": self.smoothing}, **kwargs}
        )

    def __init__(self, name, fxns, traces_kwargs=None, smoothing=True):
        """
        Supports ploteries2.figure_managers.GenericScalarsManagerViz visualizations.
        """
        self.name, self.fxns = name, [default_viz_fxn(fxn) for fxn in fxns]
        self.traces_kwargs = traces_kwargs
        self.smoothing = smoothing

    def update(self, batch, prediction, loss, timing, extra):
        # self.val=nt.copy(self.fxn(batch, prediction, loss, timing, extra))
        self.val = [
            nt.cpu(nt.detach(fxn(batch, prediction, loss, timing, extra)))
            for fxn in self.fxns
        ]

    def get(self):
        return self.val

    def write_batch(self, writer, k_sample, name=None, **name_formatters):
        name = self.get_name(name, **name_formatters)
        # getattr(writer, self._add_method)(name, self.get(), k_sample)
        writer.add_scalars(
            name,
            self.get(),
            k_sample,
            traces_kwargs=self.traces_kwargs,
            smoothing=self.smoothing,
        )


# HISTOGRAM VISUALIZATIONS


class HistogramsViz(_ScalarViz):
    def __init__(
        self,
        name,
        fxn,
        min=-np.inf,
        max=np.inf,
        trace_kwargs=None,
        layout_kwargs=None,
        **hg_kwargs,
    ):
        """
        fxn extracts a list torch tensors. A histogram will be computed from each.
        """
        self.name, self.fxn = name, default_viz_fxn(fxn)
        self.hg_kwargs = hg_kwargs
        self.bounds = np.array([min, max])
        self.add_kwargs = {
            "traces_kwargs": None if trace_kwargs is None else [trace_kwargs],
            "layout_kwargs": layout_kwargs,
        }

    def update(self, batch, prediction, loss, timing, extra):
        # self.val=nt.copy(self.fxn(batch, prediction, loss, timing, extra))
        dat = nt.detach(self.fxn(batch, prediction, loss, timing, extra))
        dat = dat.view(-1)
        if any(np.isfinite(self.bounds)):
            dat = torch.clamp(dat, self.bounds[0], self.bounds[1])
        self.val = Writer.compute_histogram(dat, **self.hg_kwargs)

    def get(self):
        return self.val  # super().get().view(-1)

    def write_batch(self, writer, k_sample, name=None, **name_formatters):
        name = self.get_name(name, **name_formatters)
        val = self.get()
        writer.add_histograms(name, [val], k_sample, **self.add_kwargs)


class HistogramsAccumViz(_AccumViz):
    def __init__(self, viz, name=None):
        """
        name: If None, the name of viz is taken.
           bin_centers are taken form the first evaluation of viz (alternatively, bin_centers can be
            explicitly provided to viz)
        """
        self.viz = viz
        self.name = name or self.viz.name
        self.normalize = self.viz.hg_kwargs.get("normalize", True)
        self.viz.hg_kwargs["normalize"] = False
        self.reset()

    def reset(self):
        self.val = None

    def update(self, true_batch_size, *args, **kwargs):
        """Computes val for current batch and updates accumulator"""
        self.viz.update(*args, **kwargs)
        bin_centers, hist = self.viz.get()
        self.viz.hg_kwargs["bin_centers"] = bin_centers
        if self.val is None:
            self.val = [bin_centers, hist]
        else:
            self.val[1] += hist

    def get(self):
        if self.val is None:
            return np.zeros(0), np.zeros(0)
        else:
            bin_centers, hist = self.val
            hist = hist.astype("f")
            if self.normalize:
                hist /= hist.sum()
            return bin_centers, hist

    def write_epoch(self, writer, k_sample, name=None, **name_formatters):
        name = self.get_name(name, **name_formatters)
        bin_centers, val = self.get()
        writer.add_histograms(
            name,
            [val],
            k_sample,
            compute_histogram=False,
            histogram_kwargs={"bin_centers": bin_centers},
            **self.viz.add_kwargs,
        )


HistogramsViz.accum_cls = HistogramsAccumViz
