from sqlalchemy import desc
from sqlalchemy.engine.result import RowProxy
import numpy as np
from pglib.py import SliceSequence
from itertools import zip_longest

import plotly.express as px
from plotly import graph_objects as go
import sqlalchemy as sqa

import colorsys
import numbers


class Colors:
    def __init__(self, name='Plotly', increase_lightness=0):
        """
        scale_lightness: [0,1]
        """
        self._rgb = list(map(lambda rgb: (rgb[0]/255, rgb[1]/255, rgb[2]/255),
                             map(px.colors.hex_to_rgb, getattr(px.colors.qualitative, name))))
        self.increase_lightness = increase_lightness

    def __len__(self):
        return len(self._rgb)

    def __getitem__(self, k):
        k = k % len(self._rgb)
        hls = colorsys.rgb_to_hls(*self._rgb[k])
        hsl = hls[0], hls[2], hls[1] + (1.0 - hls[1])*self.increase_lightness
        out = f'hsl({360*hsl[0]:.0f}, {hsl[1]:.0%}, {hsl[2]:.0%})'
        # print('**** COLORS *** : ', k, [x*255 for x in self._rgb[k]], out)
        return out
        # rgb = self._rgb[k]
        # return f'rgb({255*rgb[0]}, {255*rgb[1]}, {255*rgb[2]})'


def load_figure(reader, *args, manager_kwargs={}, **kwargs):
    """
    Loads a figure from the database and populates all its data, returning a plotly.graph_objects.Figure object.

    The kwargs can specify arguments such as the global_step.

    reader: ploteries2.reader instance
    figure: Figure id, tag or records RowProxy object.
    """

    figure_recs = reader.load_figure_recs(*args, **kwargs)
    if len(figure_recs) > 1:
        raise Exception(
            'More than one figure record matched the query specificaiton.')

    figure = figure_recs[0]

    return figure.manager(reader, **manager_kwargs).load(figure)


class FigureManager:
    widgets = tuple()  # e.g., 'slider',

    def __init__(self, reader, limit=0, sort=True, global_step=None, global_step_field='global_step'):
        """
        limit: Set to N>0 to limit the number of query outputs
        sort: Set to True to sort the query by global_step
        global_step: If set to an integer, limit and sort are ignored, and a where clause is added.
        global_step_field: Specifies the query's global_step field name.
        """
        self.reader = reader
        self.limit = limit
        self.sort = sort
        self.global_step_field = global_step_field
        self.global_step = global_step

    def process_sql(self, sql):
        if self.global_step is not None:
            sql = sql.where(f'{self.global_step_field}=self.global_step')
        else:
            if self.sort:
                sql = sql.order_by(desc(self.global_step))
            if self.limit > 0:
                sql = sql.limit(limit)
        return sql

    def _load_sql(self, sql):

        # Format query and execute
        sql = self.process_sql(sql)
        sql_output = self.reader.execute(sql)

        # Concatenate outputs to numpy arrays.
        if len(sql_output) == 0:
            return None
        else:
            sql_output = {field: np.array(
                [record[field] for record in sql_output]) for field in sql_output[0].keys()}
        return sql_output

    def load_data(self, figure_id):
        #
        # figure = self.load_figure_record(figure_tag)

        # Retrieve data maps
        datt_tbl = self.reader._data_templates
        data_templates = self.reader.execute(
            datt_tbl.select().where(datt_tbl.c.figure_id == figure_id))
        out = []
        for _dt in data_templates:
            sql_output = self._load_sql(_dt.sql)
            out.extend([(figure_slice, data_slice(sql_output))
                        for figure_slice, data_slice in _dt.data_mapper])

        #
        return out

    def load(self, figure_rec):
        data_maps = self.load_data(figure_rec.id)
        figure = figure_rec.figure

        for fig_slice, value in data_maps:
            SliceSequence(fig_slice).assign(figure, value)
        return figure


class ScalarsManager(FigureManager):

    def __init__(self, writer, n=100, **kwargs):
        super().__init__(writer, limit=0, sort=True, global_step=None, **kwargs)
        self.n = n

    def load(self, *args, **kwargs):
        #
        figure = super().load(*args, **kwargs)

        # Add smoothed curves
        if self.n > 0:
            for _dat, _smooth_dat in zip_longest(figure.data[:len(figure.data)//2],
                                                 figure.data[len(figure.data)//2:]):
                _smooth_dat.x = _dat.x
                _smooth_dat.y = self.smoothen(_dat.y)
        #
        return figure

    def smoothen(self, x):  # JS: Should pass this to javascript.
        n = self.n
        if x is None:
            return x

        def smoothing_kernel(n):
            # The nth point will contribute 1e-2 as much as the first point.
            w = np.array([1.0]) if n == 1 else np.exp(
                np.arange(n) * (np.log(1e-2)/(n-1)))
            return w/w.sum()
        w = smoothing_kernel(n)
        smoothed = np.convolve(x, w, mode='full')[:len(x)]
        # Normalize the filter in the first points.
        smoothed[:n] /= (w.cumsum())[:len(smoothed)]
        return smoothed

    @ classmethod
    def add_scalar(cls, *args, name=None, **kwargs):
        kwargs['names'] = None if name is None else [name]
        return cls.add_scalar(*args, **kwargs)

    @ classmethod
    def add_scalars(cls, writer, tag, values, global_step, names=None, **kwargs):
        #
        mode = 'lines'  # + ('+markers' if len(values) < 10 else '')

        # Cast all non-scalars to numpy array, check dtype is a real number.
        values = np.require(values)
        # if not isinstance(values, numbers.Real):
        #     values = np.require(values).reshape(1, -1)
        #     assert (np.issubdtype(values.dtype, np.number) and not np.issubdtype(
        #         values.dtype, np.complexfloating)), f'Invalid dtype {values.dtype}.'

        # Add data.
        writer.add_data(tag, values, global_step, **kwargs)

        # Register display if not done yet
        if len(writer.execute(writer._figures.select().where(writer._figures.c.tag == tag))) == 0:
            #
            colors = Colors()
            light_colors = Colors(increase_lightness=0.7)
            #
            writer.flush()
            fig = go.Figure(
                # Original data
                [go.Scatter(x=[], y=[], mode=mode, showlegend=False,
                            line=dict(color=light_colors[k]))
                 for k in range(len(values))] + \
                # Smoothed data
                [go.Scatter(x=[], y=[], mode=mode, showlegend=(names is not None),
                            line=dict(color=colors[k]), name=names[k] if names is not None else None)
                 for k in range(len(values))])
            #
            table = writer.get_data_table(tag)
            data_mappers = []
            for k in range(len(values)):
                data_mappers.append(
                    (['data', k, 'x'], ['global_step']))
                data_mappers.append(
                    (['data', k, 'y'], SliceSequence()['content'][:, k]))

            #
            writer.register_display(
                tag, fig, cls, (sqa.select([table.c.global_step, table.c.content]), data_mappers))


class PlotsManager(FigureManager):
    pass
