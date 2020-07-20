from abc import abstractmethod
from pglib.nnets import numtor
from sqlalchemy import desc, asc
from sqlalchemy.sql import func
from sqlalchemy.sql.expression import alias
from sqlalchemy.engine.result import RowProxy
import numpy as np
from pglib.py import SliceSequence
from pglib.sqlalchemy import begin_connection
from itertools import zip_longest

import plotly.express as px
from plotly import graph_objects as go
import sqlalchemy as sqa

import colorsys
import numbers

# from functools import wraps
# def register_add_method(func):
#     """
#     Signals that this method needs to be registered as an add method.
#     Prevents overwrite of add_* methods when
#     """
#     @wraps
#     def wrapper(*args, **kwargs):
#         func.registered_add_method = {'class': args[0]}
#         return func(*args, **kwargs)
#     return wrapper


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
    figure = reader.load_figure_recs(
        *args, check_single=True, **kwargs)[0]

    return figure.manager(reader, **manager_kwargs).load_figure(figure)


def global_steps(reader, *args, **kwargs):
    # Get figure record
    figure = reader.load_figure_recs(
        *args, check_single=True, **kwargs)[0]

    return figure.manager.global_steps(reader, figure)


class FigureManager:
    widgets = tuple()  # e.g., 'slider',

    @classmethod
    @abstractmethod
    def derived_table_name(cls, tag, k):
        return f'{tag}__{k}'

    @classmethod
    @abstractmethod
    def register_figure(cls, writer, tag, num_tables, connection=None, names=None):
        pass

    # @classmethod
    # def add_*(cls, writer, tag, values, global_step, *args, connection=None, **kwargs):
    #    pass

    @classmethod
    def global_steps(cls, reader, figure_rec):
        """
        Get all global steps for which the figure is available.
        """

        # Retrieve data maps
        datt_tbl = reader._data_templates
        data_templates = reader.execute(
            datt_tbl.select().where(datt_tbl.c.figure_id == figure_rec.id))

        # Carry out all queries in all data maps.
        manager = cls(reader, limit=None, global_step=None, sort='ascending')
        queries = []
        for _dt in data_templates:
            qry = manager._process_sql(_dt.sql)
            queries.append(sqa.select([qry.c.global_step]))

        # Get union of all queries to produces global steps.
        qry = sqa.union(*queries)
        qry = qry.order_by(qry.c.global_step)
        return [x[0] for x in reader.execute(sqa.select([qry.c.global_step]).distinct())]

    def __init__(self, reader, limit=None, sort='descending', global_step=None):
        """
        limit: Set to N>0 to limit the number of query outputs
        sort: Set to True to sort the query by global_step
        global_step: If set to an integer, limit and sort are ignored, and a where clause is added.
        """
        self.reader = reader
        self.limit = limit
        self.sort = {'ascending': sqa.asc,
                     'descending': sqa.desc,
                     None: None}[sort]
        self.global_step = global_step

    def _process_sql(self, qry):
        if self.global_step is not None:
            qry = alias(qry)
            qry = sqa.select([qry]).where(
                qry.c.global_step == self.global_step)
            #qry = qry.where(qry.c.global_step == self.global_step)
        else:
            if self.sort is not None:
                qry = qry.order_by(self.sort(qry.c.global_step))
            if self.limit is not None:
                qry = qry.limit(self.limit)
        return qry

    def _load_sql(self, qry, as_np_arrays=True):

        # Format query and execute
        qry = self._process_sql(qry)
        sql_output = self.reader.execute(qry)

        #
        def fuse(x): return x if not as_np_arrays else np.array(x)

        # Concatenate outputs to numpy arrays.
        if len(sql_output) == 0:
            return None
        else:
            sql_output = {field: fuse(
                [record[field] for record in sql_output]) for field in sql_output[0].keys()}
        return sql_output

    def load_data(self, figure_id, as_np_arrays=True):
        """
        as_np_arrays: Fuse all fields of all records into a single numpy array.
        """
        #
        # figure = self.load_figure_record(figure_tag)

        # Retrieve data maps
        datt_tbl = self.reader._data_templates
        data_templates = self.reader.execute(
            datt_tbl.select().where(datt_tbl.c.figure_id == figure_id))
        out = []
        for _dt in data_templates:
            sql_output = self._load_sql(_dt.sql, as_np_arrays=as_np_arrays)
            out.extend([(figure_slice, data_slice(sql_output))
                        for figure_slice, data_slice in _dt.data_mapper])

        #
        return out

    def load_figure(self, figure_rec):
        data_maps = self.load_data(figure_rec.id)
        figure = figure_rec.figure

        for fig_slice, value in data_maps:
            SliceSequence(fig_slice).assign(figure, value)
        return figure


class GenericScalarsManager(FigureManager):

    def __init__(self, writer, **kwargs):
        dflt_kwargs = dict(limit=None, sort='ascending', global_step=None)
        dflt_kwargs.update(kwargs)
        super().__init__(writer, **dflt_kwargs)

    @ classmethod
    def build_figure_template(cls, num_scalars, names, colors=Colors(), trace_kwargs=None):
        """
        names=( None | [name1, name2, None, name4])
        trace_kwargs = ( None | [{kws1}, {kws2}, None, {kws4}] | {kws} )
        """

        #
        mode = 'lines'
        #
        fig = go.Figure(
            [go.Scatter(x=[], y=[], mode=mode, showlegend=(names is not None and names[k] is not None),
                        line=dict(color=colors[k]), name=names[k] if names is not None else None,
                        **({} if trace_kwargs is None else
                           trace_kwargs if isinstance(trace_kwargs, dict) else
                           trace_kwargs[k]))
             for k in range(num_scalars)])
        #
        return fig

    @ classmethod
    def register_figure(cls, writer, tag, num_scalars, names=None, connection=None, trace_kwargs=None, **kwargs):
        writer.flush()
        fig = cls.build_figure_template(
            num_scalars, names=names, trace_kwargs=trace_kwargs, **kwargs)
        #
        data_mappers = []
        for k in range(num_scalars):
            data_mappers.append(
                (['data', k, 'x'], ['global_step']))
            data_mappers.append(
                (['data', k, 'y'], SliceSequence()['content'][:, k]))

        # Create data table and add figure record to database
        with begin_connection(writer.engine, connection) as conn:
            writer.create_data_table(tag, np.ndarray, connection=conn)
            table = writer.get_data_table(tag)
            writer.register_figure(
                tag, fig, cls, (sqa.select([table.c.global_step, table.c.content]), data_mappers), connection=conn)

    @ classmethod
    def add_generic_scalars(
            cls, writer, tag, values, global_step, names=None, connection=None, write_time=None, trace_kwargs=None):

        # Cast all non-scalars to numpy array, check dtype is a real number.
        values = np.require(values)
        if not (values.ndim <= 1 and np.issubdtype(values.dtype, np.number) and not np.issubdtype(
                values.dtype, np.complexfloating)):
            raise Exception(f'Invalid dtype {values.dtype}.')
        if values.ndim == 0:
            values.shape = (1,)

        with begin_connection(writer.engine, connection) as conn:
            # Register figure if not done yet
            if not writer.figure_exists(tag, {'manager': cls}):
                cls.register_figure(writer, tag, len(
                    values), names=names, connection=conn, trace_kwargs=trace_kwargs)

            # Add data.
            writer.add_data(tag, values, global_step,
                            write_time=write_time, create=False)


class SmoothenedScalarsManager(GenericScalarsManager):

    def __init__(self, writer, smoothing_n=100, **kwargs):
        super().__init__(writer, **kwargs)
        self.smoothing_n = smoothing_n

    def load_figure(self, *args, **kwargs):
        #
        figure = super().load_figure(*args, **kwargs)

        # Add smoothed curves
        if self.smoothing_n > 0:
            for _dat, _smooth_dat in zip_longest(figure.data[:len(figure.data)//2],
                                                 figure.data[len(figure.data)//2:]):
                _smooth_dat.x = _dat.x
                _smooth_dat.y = self.smoothen(_dat.y)
        #
        return figure

    def smoothen(self, x):  # TODO: Should pass this to javascript.
        smoothing_n = self.smoothing_n
        if x is None:
            return x

        def smoothing_kernel(smoothing_n):
            # The nth point will contribute 1e-2 as much as the first point.
            w = np.array([1.0]) if smoothing_n == 1 else np.exp(
                np.arange(smoothing_n) * (np.log(1e-2)/(smoothing_n-1)))
            return w/w.sum()
        w = smoothing_kernel(smoothing_n)
        smoothed = np.convolve(x, w, mode='full')[:len(x)]
        # Normalize the filter in the first points.
        smoothed[:smoothing_n] /= (w.cumsum())[:len(smoothed)]
        return smoothed

    @ classmethod
    def build_figure_template(cls, num_scalars, names, trace_kwargs=None):
        #
        colors = Colors()
        light_colors = Colors(increase_lightness=0.7)
        #
        all_colors = ([light_colors[k] for k in range(num_scalars)] +
                      [colors[k] for k in range(num_scalars)])
        all_names = None if names is None else ([None]*num_scalars + names)
        #
        fig = super().build_figure_template(
            2*num_scalars, all_names, all_colors, trace_kwargs=trace_kwargs)
        #
        return fig

    @ classmethod
    def add_scalar(cls, *args, name=None, **kwargs):
        kwargs['names'] = None if name is None else [name]
        return cls.add_scalars(*args, **kwargs)

    @ classmethod
    def add_scalars(cls, writer, tag, values, global_step, names=None, connection=None, write_time=None):
        #
        super().add_generic_scalars(
            writer, tag, values, global_step, names=None, connection=None, write_time=None)


# class ScalarsManager_old(FigureManager):

#     def __init__(self, writer, smoothing_n=100, **kwargs):
#         dflt_kwargs = dict(limit=None, sort='ascending', global_step=None)
#         dflt_kwargs.update(kwargs)
#         super().__init__(writer, **dflt_kwargs)
#         self.smoothing_n = smoothing_n

#     def load_figure(self, *args, **kwargs):
#         #
#         figure = super().load_figure(*args, **kwargs)

#         # Add smoothed curves
#         if self.smoothing_n > 0:
#             for _dat, _smooth_dat in zip_longest(figure.data[:len(figure.data)//2],
#                                                  figure.data[len(figure.data)//2:]):
#                 _smooth_dat.x = _dat.x
#                 _smooth_dat.y = self.smoothen(_dat.y)
#         #
#         return figure

#     def smoothen(self, x):  # TODO: Should pass this to javascript.
#         smoothing_n = self.smoothing_n
#         if x is None:
#             return x

#         def smoothing_kernel(smoothing_n):
#             # The nth point will contribute 1e-2 as much as the first point.
#             w = np.array([1.0]) if smoothing_n == 1 else np.exp(
#                 np.arange(smoothing_n) * (np.log(1e-2)/(smoothing_n-1)))
#             return w/w.sum()
#         w = smoothing_kernel(smoothing_n)
#         smoothed = np.convolve(x, w, mode='full')[:len(x)]
#         # Normalize the filter in the first points.
#         smoothed[:smoothing_n] /= (w.cumsum())[:len(smoothed)]
#         return smoothed

#     @ classmethod
#     def build_figure_template(cls, num_scalars, names):
#         #
#         mode = 'lines'  # + ('+markers' if len(values) < 10 else '')
#         colors = Colors()
#         light_colors = Colors(increase_lightness=0.7)
#         #
#         fig = go.Figure(
#             # Original data
#             [go.Scatter(x=[], y=[], mode=mode, showlegend=False,
#                         line=dict(color=light_colors[k]))
#              for k in range(num_scalars)] + \
#             # Smoothed data
#             [go.Scatter(x=[], y=[], mode=mode, showlegend=(names is not None and names[k] is not None),
#                         line=dict(color=colors[k]), name=names[k] if names is not None else None)
#              for k in range(num_scalars)])
#         #
#         return fig

#     @ classmethod
#     def register_figure(cls, writer, tag, num_scalars, names=None, connection=None):
#         writer.flush()
#         fig = cls.build_figure_template(num_scalars, names=names)
#         #
#         data_mappers = []
#         for k in range(num_scalars):
#             data_mappers.append(
#                 (['data', k, 'x'], ['global_step']))
#             data_mappers.append(
#                 (['data', k, 'y'], SliceSequence()['content'][:, k]))

#         # Create data table and add figure record to database
#         with begin_connection(writer.engine, connection) as conn:
#             writer.create_data_table(tag, np.ndarray, connection=conn)
#             table = writer.get_data_table(tag)
#             writer.register_figure(
#                 tag, fig, cls, (sqa.select([table.c.global_step, table.c.content]), data_mappers), connection=conn)

#     @ classmethod
#     def add_scalar(cls, *args, name=None, **kwargs):
#         kwargs['names'] = None if name is None else [name]
#         return cls.add_scalars(*args, **kwargs)

#     @ classmethod
#     def add_scalars(cls, writer, tag, values, global_step, names=None, connection=None, write_time=None):
#         #

#         # Cast all non-scalars to numpy array, check dtype is a real number.
#         values = np.require(values)
#         if not (values.ndim <= 1 and np.issubdtype(values.dtype, np.number) and not np.issubdtype(
#                 values.dtype, np.complexfloating)):
#             raise Exception(f'Invalid dtype {values.dtype}.')
#         if values.ndim == 0:
#             values.shape = (1,)

#         with begin_connection(writer.engine, connection) as conn:
#             # Register figure if not done yet
#             if not writer.figure_exists(tag, {'manager': cls}):
#                 cls.register_figure(writer, tag, len(
#                     values), names=names, connection=conn)

#             # Add data.
#             writer.add_data(tag, values, global_step,
#                             write_time=write_time, create=False)


class PlotsManager(FigureManager):

    def __init__(self, writer, global_step=None, **kwargs):
        default_kwargs = {
            'limit': 1, 'sort': 'descending'}
        default_kwargs.update(kwargs)
        super().__init__(
            writer, global_step=global_step, **default_kwargs)

    @ classmethod
    def derived_table_name(cls, tag, k):
        return f'{tag}__{k}'

    @ classmethod
    def build_figure_template(cls, num_traces, names):
        mode = 'lines'  # + ('+markers' if len(values) < 10 else '')
        colors = Colors()
        #
        fig = go.Figure(
            [go.Scatter(x=[], y=[], mode=mode, showlegend=(names is not None and names[k] is not None),
                        line=dict(color=colors[k]), name=names[k] if names is not None else None)
             for k in range(num_traces)])
        #
        return fig

    @classmethod
    def register_figure(cls, writer, tag, num_tables, connection=None, names=None, _test_exceptions=[]):
        # Register display if not done yet
        ###################################
        # Create data tables
        with begin_connection(writer.engine, connection) as conn:
            #
            [writer.create_data_table(
                cls.derived_table_name(tag, _k), np.ndarray, connection=conn, indexed_global_step=True)
             for _k in range(num_tables)]
            if 'post_create_tables' in _test_exceptions:
                raise Exception('post_create_tables')
            # writer.flush()

            #
            fig = cls.build_figure_template(num_tables, names)

            #
            data_mappers = []
            for k in range(num_tables):
                data_mappers.append(
                    (['data', k, f'x'], [f'content{k}', 0, 0]))
                data_mappers.append(
                    (['data', k, f'y'], [f'content{k}', 0, 1]))

            # Build sql outer joins across all tables, keep all global_step values, even when not present in all tables.
            tables = [(writer.get_data_table(cls.derived_table_name(tag, k)), f'content{k}')
                      for k in range(num_tables)]
            if 'pre_sql' in _test_exceptions:
                raise Exception('pre_sql')
            sql = writer.outer_join_data_tables(tables)
            # sql = alias(sqa.select(
            #     [tables[0].c.global_step, tables[0].c.content]))
            # content_cols = [sql.c.content.label('content0')]
            # for k, curr_table in enumerate(tables[1:]):
            #     sql = reader.outer_join_data_tables
            #     sql = alias(sqa.select([
            #         func.ifnull(sql.c.global_step, curr_table.c.global_step).label(
            #             'global_step'),
            #         curr_table.c.content]).select_from(
            #             sql.outerjoin(curr_table, curr_table.c.global_step == sql.c.global_step)))
            #     content_cols.append(
            #         sql.c.content.label(f'content{k+1}'))

            # sql = sqa.select([sql.c.global_step.label('global_step')] +
            #                  content_cols).select_from(sql)

            writer.register_figure(
                tag, fig, cls, (sql, data_mappers), connection=conn)

    @ classmethod
    def add_plots(cls, writer, tag, values, global_step, names=None, connection=None, write_time=None):
        """
        values: List of entities that can each be converted to np.ndarrays of 2 rows (x and y) and arbitrary columns.
        write_time: passed to writer.add_data function
        """

        # Get and verify numpy arrays.
        values = [np.require(_v) for _v in values]
        if not all((_v.ndim == 2 and _v.shape[0] == 2 for _v in values)):
            raise Exception('Invalid input value shapes.')

        with begin_connection(writer.engine, connection) as conn:
            # Register figure if it does not exist yet, create all related tables atomically.
            if not writer.figure_exists(tag, {'manager': cls}):
                cls.register_figure(writer, tag,
                                    len(values), connection=conn, names=names)

        # Add data.
        # ##################
        [writer.add_data(
            cls.derived_table_name(tag, _k), _v, global_step, connection=conn, write_time=write_time, create=False)
         for _k, _v in enumerate(values)]


class HistogramsManager(PlotsManager):
    @ classmethod
    def build_figure_template(cls, num_traces, names):
        colors = Colors()
        light_colors = Colors(increase_lightness=0.7)
        #
        fig = go.Figure(
            [go.Bar(x=[], y=[], showlegend=(names is not None and names[k] is not None),
                    marker_color=light_colors[k], marker_line_color=colors[k], marker_line_width=1.5, opacity=0.7,
                    name=names[k] if names is not None else None)
             for k in range(num_traces)])
        fig.update_layout(bargap=0.1, bargroupgap=0.05, barmode='group')
        #
        return fig

    @ classmethod
    def add_histograms(cls, writer, tag, values, global_step, names=None, connection=None, write_time=None, **histo_kwargs):
        """
        values: list of entities convertible to np.ndarray of 1 dimension.
        """

        # Compute histogram.
        dat = np.concatenate(values)
        if dat.ndim != 1:
            raise Exception('Invalid input shapes.')
        bin_centers, hist = cls._compute_histogram(dat, **histo_kwargs)
        plot_values = [cls._compute_histogram(_val, bin_centers=bin_centers)
                       for _val in values]

        # Add figure data
        cls.add_plots(
            writer, tag, plot_values, global_step, names=names, connection=connection, write_time=write_time)

    @ staticmethod
    def _compute_histogram(dat, bins=20, bin_centers=None, normalize=True):
        """
        'bins': Num bins or bin edges (passed to numpy.histogram to create a histogram).
        'bin_centers': Overrides 'bins' and specifies the bin centers instead of the edges.
            The first and last bin centers are assumed to extend to +/- infinity.
        """
        dat = numtor.asnumpy(dat)
        dat = dat.reshape(-1)

        # Infer bin edges
        if bin_centers is not None:
            bins = np.block(
                [-np.inf, np.convolve(bin_centers, [0.5, 0.5], mode='valid'), np.inf])

        # Build histogram
        hist, edges = np.histogram(dat, bins=bins)

        # Infer bin centers
        if bin_centers is None:
            bin_centers = np.convolve(edges, [0.5, 0.5], mode='valid')
            for k in [0, -1]:
                if not np.isfinite(bin_centers[k]):
                    bin_centers[k] = edges[k]

        # Normalize histogram
        if normalize:
            hist = hist/dat.size

        return bin_centers, hist
