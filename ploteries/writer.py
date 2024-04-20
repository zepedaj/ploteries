"""
Provides a higher-level interface to ploteries that exposes a :class:`Writer` class with :meth:`add_*` methods similar to Tensorboard's API.
"""

import numpy as np
from numbers import Number

from .data_store import DataStore, Ref_
from .ndarray_data_handlers import UniformNDArrayDataHandler, RaggedNDArrayDataHandler
from .serializable_data_handler import SerializableDataHandler
from .figure_handler import FigureHandler, TableHandler
from jztools.numpy import ArrayLike
from typing import Optional, List, Dict, Any, Type
from jztools.nnets import numtor
import os.path as osp
from .figure_handler.scalars_figure_handler import ScalarsFigureHandler


class Writer:
    """
        All :meth:`add_*` methods build a list of trace dictionaries by combining corresponding dictionaries derived from the argument iterable :attr:`values` and the list of dictionaries in the argument :attr:`traces_kwargs`. The :attr:`values` iterable is expected to change from one time index to the next and its contents are stored for each time index in the data records table using a name tag derived from the :meth:`add_*` methods's name or specified using argument :attr:`data_name`. The :attr:`traces_kwargs`, however, will be saved along with :attr:`layout_kwargs` in a figure template that is stored in the figure definitions table. Note that this figure template is only built the first time the :attr:`add_*` method is called, and that arguments :attr:`traces_kwargs` and :attr:`layout_kwargs` are ignored in subsequent calls.

    `    .. todo:: We need to 1) check that figure-definition values are compatible with the existing figure; 2) add mode="overwrite" functionality. It might be best to make each add_* function a class deriving from a BaseAdder class.
    """

    _table_names = {
        "add_scalars": "__add_scalars__.{figure_name}",
        "add_plots": "__add_plots__.{figure_name}",
        "add_histograms": "__add_histograms__.{figure_name}",
        "add_table": "__add_table__.{figure_name}",
    }

    def __init__(self, path):
        """
        :param path: Data store directory or file path. If a directory is specified, ``'data_store.pltr'`` will be appended.
        """
        if osp.isdir(path):
            path = osp.join(path, "ploteries.pltr")
        self.data_store = path if isinstance(path, DataStore) else DataStore(path)

        # Keep a cache of existing figures and data handlers
        # to avoid saturating the database with requests.
        self.existing_figures = set()
        self.existing_data_handlers = {}

    def flush(self):
        self.data_store.flush()

    @classmethod
    def _get_table_name(cls, func, **kwargs):
        return cls._table_names[func].format(**kwargs)

    def _get_data_handler(self, data_name, func):
        if (data_handler := self.existing_data_handlers.get(data_name, None)) is None:
            self.existing_data_handlers[data_name] = (data_handler := func())
        return data_handler

    def _write_figure(
        self,
        figure_name,
        value_refs_as_traces: List[Dict[str, Ref_]],
        traces_kwargs: List[Dict[str, Any]],
        default_trace_kwargs: Dict[str, Any],
        layout_kwargs: Dict[str, Any],
        figure_handler: Type[FigureHandler] = FigureHandler,
    ):
        """
        Checks that the lengths of value_refs_as_traces and number of traces_kwargs match. Writes the figure to the figure definitions table if the figure does not yet exist. Returns ``True`` if the figure was written and ``False`` otherwise. Trace keyword args will be built by combining corresponding dictionaries in value_refs_as_traces and traces_kwargs, and applying to each the dictionary in default_trace_kwargs. Key collisions are resolved in the following highest-to-lowest priority order: ``value_refs_as_traces``, ``traces_kwargs``, ``default_trace_kwargs``.

        :param value_refs_as_traces: List of trace dictionaries containing :class:`Ref_` data placeholders.
        """
        figure_written = False
        if figure_name not in self.existing_figures:
            # Check traces_kwargs input
            if traces_kwargs and len(traces_kwargs) != len(value_refs_as_traces):
                raise ValueError(
                    f"Param traces_kwargs has {len(traces_kwargs)} values, "
                    f"but expected 0 or {len(value_refs_as_traces)}."
                )
            traces_kwargs = traces_kwargs or [{}] * len(value_refs_as_traces)

            # Add trace kwargs
            value_refs_as_traces = [
                {**_trace_kwargs, **_trace}
                for _trace, _trace_kwargs in zip(value_refs_as_traces, traces_kwargs)
            ]

            # Create figure handler.
            fig_handler = figure_handler.from_traces(
                self.data_store,
                name=figure_name,
                traces=value_refs_as_traces,
                default_trace_kwargs=default_trace_kwargs,
                layout_kwargs=layout_kwargs,
            )

            # Write figure (if it does not exist).
            figure_written = fig_handler.write_def()
            self.existing_figures.add(figure_name)

        #
        return figure_written

    def add_scalar(self, *arg, **kwargs):
        """
        Alias for :meth:`add_scalars`.
        """
        return self.add_scalars(*arg, **kwargs)

    def add_scalars(
        self,
        figure_name: str,
        values: ArrayLike,
        global_step: int,
        traces_kwargs: Optional[List[Dict]] = None,
        layout_kwargs=None,
        data_name: Optional[str] = None,
        smoothing: bool = True,
    ):
        """
        :param figure_name: The figure name. If specified in format '<tab>/<group>/...' , the tab and group entries will determine the position of the figure in the page.
        :param values: The values for each scalar trace as an array-like.
        :param names: The legend name to use for each trace.
        :param traces: None or list of dictionaries of length equal to that of values containing keyword arguments for the trace. The default value for each trace is ``{'type':'scatter', 'mode':'lines'}`` and will be updated with the specified values.
        :param data_name: The name of the data series when stored in the data table. If ``None``, the name ``__add_scalars__.<figure_name>`` will be used.
        :param smoothing: Whether to enable smoothing.

        Example:

        ```
        writer.add_scalars(
            'three_plots', [0.1, 0.4, 0.5],
            10,
            [{'type': 'scatter', 'name': 'trace 0'},
             {'name': 'trace 1'},
             {'type': 'bar', 'name': 'trace 2'}])
        ```
        """
        layout_kwargs = layout_kwargs or {}
        default_trace_kwargs = {"type": "scatter", "mode": "lines"}

        # Get data name.
        data_name = data_name or self._get_table_name(
            "add_scalars", figure_name=figure_name
        )

        # Check values input
        values = numtor.asnumpy(values)
        values = values[None] if values.ndim == 0 else values
        if values.ndim != 1 or not isinstance(values[0], Number):
            raise ValueError(
                f"Expected a 1-dim array-like object of numbers, but obtained an "
                f"array of shape {values.shape} with {values.dtype} entries."
            )

        # Write figure def
        if figure_name not in self.existing_figures:
            traces = [
                {
                    "x": Ref_(data_name)["meta"]["index"],
                    "y": Ref_(data_name)["data"][:, k],
                }
                for k in range(len(values))
            ]
            self._write_figure(
                figure_name,
                traces,
                traces_kwargs,
                default_trace_kwargs,
                layout_kwargs,
                figure_handler=(ScalarsFigureHandler if smoothing else FigureHandler),
            )

        # Write data.
        data_handler = self._get_data_handler(
            data_name,
            lambda: UniformNDArrayDataHandler(self.data_store, name=data_name),
        )
        data_handler.add_data(global_step, values)

    def add_plots(
        self,
        figure_name: str,
        values: ArrayLike,
        global_step: int,
        traces_kwargs: Optional[List[Dict]] = None,
        layout_kwargs=None,
        data_name: Optional[str] = None,
    ):
        """
        :param figure_name: (See :meth:`add_scalars`).
        :param values: The values for each scalar trace as a dictionary, e.g., ``[{'x': [0,2,4], 'y': [0,2,4]}, {'x': [0,2,4], 'y': [0,4,16]}]``. Dictionaries can contain lists, strings numpy ndarrays and generally anything compatible with :class:`~xerializer.Serializer`. These dictionaries will be udpated with the corresponding value of traces_kwargs, if any. Note that the content of values will change with the global step and is saved in the data records table, but the content of traces_kwargs will remain constant and stored with the figure definition.
        :param data_name: (See :meth:`add_scalars`).
        :param traces_kwargs: (See :meth:`add_scalars`).
        :param layout_kwargs: (See :meth:`add_scalars`).

        Example:

        ```
        writer.add_plots(
            'three_plots', [{'x': [0,2,4], 'y': [0,2,4]}, {'x': [0,2,4], 'y': [0,4,16]}],
            10,
            [{'type': 'scatter', 'name': 'trace 0'},
             {'name': 'trace 1'},
             {'type': 'bar', 'name': 'trace 2'}])
        ```
        """

        layout_kwargs = layout_kwargs or {}
        default_trace_kwargs = {"type": "scatter", "mode": "lines"}

        # Get data name.
        data_name = data_name or self._get_table_name(
            "add_plots", figure_name=figure_name
        )

        # Write figure def.
        if figure_name not in self.existing_figures:
            # Build traces with data store references.
            traces = [
                {
                    _key: Ref_({"data": data_name, "index": "latest"})["data"][0][k][
                        _key
                    ]
                    for _key in values[k]
                }
                for k in range(len(values))
            ]
            self._write_figure(
                figure_name, traces, traces_kwargs, default_trace_kwargs, layout_kwargs
            )

        # Write data
        data_handler = self._get_data_handler(
            data_name, lambda: SerializableDataHandler(self.data_store, name=data_name)
        )
        data_handler.add_data(global_step, values)

    def add_histograms(
        self,
        figure_name: str,
        values: List[ArrayLike],
        global_step: int,
        traces_kwargs: Optional[List[Dict]] = None,
        layout_kwargs=None,
        data_name: Optional[str] = None,
        compute_histogram: bool = True,
        histogram_kwargs: dict = None,
    ):
        """
        :param figure_name: (See :meth:`add_scalars`).
        :param values: A list of 1-row array-like entries.
            * When :attr:`compute_histogram` is ``True`` (the default), bin centers will be computed from all entries together, taking param :attr:`histogram_kwargs` into account.
            * When :attr:`compute_histogram` is ``False`` (requires an entry 'bin_centers' in param :attr:`histogram_kwargs`), each entry will be assumed to be a pre-computed histogram.
        :param data_name: (See :meth:`add_scalars`).
        :param traces_kwargs: (See :meth:`add_scalars`).
        :param <layout_kwargs: (See :meth:`add_scalars`).
        :param compute_histogram: (See param :attr:`values`.). Note that this param can change between different calls with the same figure_name.
        :param histogram_kwargs: When :attr:`compute_histogram` is ``True``, these kwargs will be passed to :meth:`compute_histogram`. Note that this param can change between different calls with the same figure_name.

        Example:

        ```
        """

        layout_kwargs = layout_kwargs or {}
        histogram_kwargs = histogram_kwargs or {}
        default_trace_kwargs = {
            "type": "bar",
            # 'marker_line_width': 1.5,
            # 'opacity': 0.7
        }
        layout_kwargs = {
            # **{'bargap': 0.01, 'bargroupgap': 0.05, 'barmode': 'group'},
            **layout_kwargs
        }

        # Get data name.
        data_name = data_name or self._get_table_name(
            "add_histograms", figure_name=figure_name
        )

        # Check values
        values = [numtor.asnumpy(_x).reshape(-1) for _x in values]
        if not compute_histogram and "bin_centers" not in histogram_kwargs:
            raise Exception(
                "When compute_histogram=False, histogram_kwargs needs to contain a key "
                "'bin_centers' with the bin centers."
            )

        def fld_(k):
            return f"field_{k}"

        # Write figure def.
        if figure_name not in self.existing_figures:
            # Build traces with data store references.
            traces = [
                {
                    "x": Ref_({"data": data_name, "index": "latest"})["data"][0][
                        "bin_centers"
                    ],
                    "y": Ref_({"data": data_name, "index": "latest"})["data"][0][
                        fld_(_k)
                    ],
                }
                for _k in range(len(values))
            ]
            self._write_figure(
                figure_name, traces, traces_kwargs, default_trace_kwargs, layout_kwargs
            )

        # Compute histogram.
        if "bin_centers" not in histogram_kwargs:
            histogram_kwargs["bin_centers"] = self.compute_histogram(
                np.concatenate(values)
            )[0]
        if compute_histogram:
            values = [
                np.array(self.compute_histogram(_v, **histogram_kwargs))[1]
                for _v in values
            ]

        # Assemble data to write
        bin_centers = histogram_kwargs["bin_centers"]
        values_array = np.empty(
            len(bin_centers),
            dtype=(
                [("bin_centers", bin_centers.dtype)]
                + [(fld_(_k), _val.dtype) for _k, _val in enumerate(values)]
            ),
        )
        values_array["bin_centers"] = bin_centers
        for _k, _val in enumerate(values):
            values_array[f"field_{_k}"][:] = _val

        # Write data
        data_handler = self._get_data_handler(
            data_name, lambda: RaggedNDArrayDataHandler(self.data_store, name=data_name)
        )
        data_handler.add_data(global_step, values_array)

    @staticmethod
    def compute_histogram(dat, bins=20, bin_centers=None, normalize=True):
        """
        :param dat: Array-like from which the histogram will be computed.
        :param bins: Num bins or bin edges (passed to numpy.histogram to create a histogram).
        :param bin_centers: Overrides 'bins' and specifies the bin centers instead of the edges.
            The first and last bin centers are assumed to extend to +/- infinity.
        :param normalize: Normalize the histogram so that it adds up to one.
        """
        dat = numtor.asnumpy(dat)
        dat = dat.reshape(-1)

        # Infer bin edges
        if bin_centers is not None:
            bins = np.block(
                [-np.inf, np.convolve(bin_centers, [0.5, 0.5], mode="valid"), np.inf]
            )

        # Build histogram
        hist, edges = np.histogram(dat, bins=bins)

        # Infer bin centers
        if bin_centers is None:
            bin_centers = np.convolve(edges, [0.5, 0.5], mode="valid")
            for k in [0, -1]:
                if not np.isfinite(bin_centers[k]):
                    bin_centers[k] = edges[k]

        # Normalize histogram
        if normalize:
            hist = hist / dat.size

        return bin_centers, hist

    def add_table(
        self,
        figure_name: str,
        values: dict,
        global_step: int,
        data_name: Optional[str] = None,
        **kwargs,
    ):
        """
        :param figure_name: (See :meth:`add_scalars`).
        :param values: A dictionary of values. Each key in the dictionary will define a column (row, if transposed=True) in the table, and the corresponding value will contain the row (resp., column) values for the specified global_step.
        :param data_name: (See :meth:`add_scalars`).
        :param **kwargs: Extra arguments for class : class: `ploteries.figure_handlers.TableHandler`.

        Example:

        ```
        writer.add_table(
            'my_table', {'col1': 1.0, 'col2': 2.0}, 10)
        ```
        """

        # Get data name.
        data_name = data_name or self._get_table_name(
            "add_table", figure_name=figure_name
        )

        # Write figure def.
        if figure_name not in self.existing_figures:
            # Build traces with data store references.
            tbl_h = TableHandler(
                self.data_store, figure_name, (data_name, []), **kwargs
            )

            tbl_h.write_def()
            self.existing_figures.add(figure_name)

        # Write data
        data_handler = self._get_data_handler(
            data_name, lambda: SerializableDataHandler(self.data_store, name=data_name)
        )
        data_handler.add_data(global_step, values)
