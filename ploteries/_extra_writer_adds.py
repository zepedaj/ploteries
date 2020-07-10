import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def add_line_plot(writer, tag, data, global_step):
    """
    Add plot with multiple traces and potentially double x or y (or both) axes.

    data: Dictionary with following fields:
    'title', 'xtitle', 'ytitle': Optional, can be string or 2-tuples of strings for double x/y axes.
    'data': Iterable with each value containing a dictionary of the form 
       {'x':<iterable>, 'y':<iterable>, ['name':<string], ['secondary':('y' | 'x' | 'xy' | 'yx')]}
    (x/y)title: Use 2-tuples for secondary titles, or string for single-title.
    """

    values = data['data']
    title, xtitle, ytitle = [data.get(_key, None)
                             for _key in ['title', 'xtitle', 'ytitle']]

    def check_secondary(x_or_y, _v):
        return _v.get('secondary', None) in [x_or_y, 'xy', 'yx']

    # Check if secondary y.
    secondary_x = (any((check_secondary('x', _v) for _v in values)) or
                   (isinstance(xtitle, (tuple, list)) and len(xtitle) == 2))
    secondary_y = (any((check_secondary('y', _v) for _v in values)) or
                   (isinstance(ytitle, (tuple, list)) and len(ytitle) == 2))

    # Create figure with secondary axes
    fig = make_subplots(
        specs=[[{"secondary_y": secondary_y}]])
    # specs=[[{"secondary_y": secondary_y, 'secondary_x': secondary_x}]])

    # Add traces
    for _v in values:
        # Get secondary axes options.
        trace_kwargs = {}
        if secondary_y:
            trace_kwargs.update({'secondary_y': check_secondary('y', _v)}),
        if secondary_x:
            trace_kwargs.update({'secondary_x': check_secondary('x', _v)}),

        # Add traces
        fig.add_trace(go.Scatter(
            x=_v['x'], y=_v['y'], name=_v.get('name', None)), **trace_kwargs)

    # Add figure title
    if title:
        fig.update_layout(title_text=title)

    # Add axes titles
    def set_axes_titles(x_or_y, title):
        #
        if title is None:
            return
        #
        secondary = secondary_y  # globals()[f'secondary_{x_or_y}']
        update_fxn = getattr(fig, f'update_{x_or_y}axes')
        #
        if isinstance(title, str) and not secondary:
            update_fxn(title_text=title)
        elif isinstance(title, str) and secondary:
            update_fxn(title_text=title, **{f'secondary_{x_or_y}': False})
        elif isinstance(title, (tuple, list)) and len(title) == 2 and secondary:
            update_fxn(title_text=None, **{f'secondary_{x_or_y}': False})
            update_fxn(title_text=title[1], **{f'secondary_{x_or_y}': True})
        else:
            raise Exception('Invalid input.')
    #
    if not title is None:
        fig.update_layout(title_text=title)
    set_axes_titles('x', xtitle)
    set_axes_titles('y', ytitle)

    # fig.show()
    if writer is not None:
        writer.add_figure(tag, fig, global_step)
    return fig


# Test


class Writer:
    def add_figure(*args):
        pass


fig = add_line_plot(
    Writer(), 'plot_name',
    {'data': [
        {'x': np.arange(10), 'y': np.arange(10)},
        {'x': np.arange(20, 30), 'y': np.arange(20, 30), 'secondary': 'y'}],
     'ytitle': ('abc', 'def'), }, 0)
print(fig)
