from .figure_handler import FigureHandler
from plotly.subplots import make_subplots
from typing import List
from ..data_store import DataStore


def plotyy(
    data_store: DataStore,
    name: str,
    primary_traces: List[dict],
    secondary_traces: List[dict],
    default_trace_kwargs={},
    layout_kwargs={},
    connection=None,
    write=True,
):
    """
    Creates a figure with a primary and secondary y axis

    :param data_store: Data store where the figure will be written.
    :param name: Figure name in the data store.
    :param primary_traces, secondary_traces: The traces for the primary and secondary axes.
    :param default_trace_kwargs, layout_kwargs, connection: These parameters will be passed on to :meth:`~ploteries.figure_handler.figure_handler.FigureHandler.from_traces`
    """

    # Get layout kwargs
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.layout.template = None
    layout_kwargs = {**fig.to_dict()["layout"], **layout_kwargs}

    # Update traces
    primary_traces = [{**{"xaxis": "x", "yaxis": "y"}, **_tr} for _tr in primary_traces]
    secondary_traces = [
        {**{"xaxis": "x", "yaxis": "y2"}, **_tr} for _tr in secondary_traces
    ]

    # Build figure handler
    fig_h = FigureHandler.from_traces(
        data_store,
        name,
        primary_traces + secondary_traces,
        default_trace_kwargs=default_trace_kwargs,
        layout_kwargs=layout_kwargs,
        connection=connection,
    )

    return fig_h
