from sqlalchemy import desc
import numpy as np
from pglib.py import SliceSequence
from itertools import zip_longest

import plotly.express as px
from plotly import graph_objects as go
import sqlalchemy as sqa


class FigureManager:
    def __init__(self, writer, limit=0, sort=True, global_step=None, global_step_field='global_step'):
        """
        limit: Set to N>0 to limit the number of query outputs
        sort: Set to True to sort the query by global_step
        global_step: If set to an integer, limit and sort are ignored, and a where clause is added.
        global_step_field: Specifies the query's global_step field name.
        """
        self.writer = writer
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
        sql_output = self.writer.execute(sql)

        # Concatenate outputs to numpy arrays.
        if len(sql_output) == 0:
            return None
        elif len(sql_output) > 1:
            sql_output = {field: np.array(
                [record[field] for record in sql_output]) for field in sql_output[0].keys()}
        return sql_output

    def load_data(self, figure_tag):

        # Retrieve figure data
        figs_tbl = self.writer._figures
        figure = self.writer.execute(
            self.writer._figures.select().where(figs_tbl.c.tag == figure_tag))[0]

        # Retrieve data maps
        datt_tbl = self.writer._data_templates
        data_templates = self.writer.execute(
            datt_tbl.select().where(datt_tbl.c.figure_id == figure.id))
        out = []
        for _dt in data_templates:
            sql_output = self._load_sql(_dt.sql)
            out.extend([(figure_slice, data_slice(sql_output))
                        for figure_slice, data_slice in _dt.data_mapper])

        #
        return figure.figure, out

    def load(self, figure_tag):
        figure, data_maps = self.load_data(figure_tag)
        for fig_slice, value in data_maps:
            SliceSequence(fig_slice).assign(figure, value)
        return figure


class ScalarsManager(FigureManager):
    register = ['add_scalars', 'add_scalar']

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

    @staticmethod
    def add_scalar(*args, **kwargs):
        return self.add_scalar(*args, **kwargs)

    @staticmethod
    def add_scalars(writer, tag, values, global_step, **kwargs):
        #
        mode = 'lines'  # + ('+markers' if len(values) < 10 else '')

        if not isinstance(values, np.ndarray):
            values = np.ndarray(values).reshape(1, -1)

        # Add data.
        writer.add_data(tag, values, global_step, **kwargs)

        # Register display if not done yet
        if len(writer.execute(writer._figures.select().where(writer._figures.c.tag == tag))) == 0:
            #
            writer.flush()
            fig = go.Figure(
                # Original data
                [go.Scatter(x=[], y=[], mode=mode, line=dict(
                    color='hsl(217, 59%, 80%)'))] * len(values) + \
                # Smoothed data
                [go.Scatter(x=[], y=[], mode=mode, line=dict(
                    color='hsl(217, 59%, 50%)'))] * len(values))
            #
            table = writer.get_table(tag)
            data_mappers = []
            for k in range(len(values)):
                data_mappers.append(
                    (['data', k, 'x'], ['global_step']))
                data_mappers.append(
                    (['data', k, 'y'], SliceSequence()['content'][:, k]))

            #
            writer.register_display(
                tag, fig, (sqa.select([table.c.global_step, table.c.content]), data_mappers))
