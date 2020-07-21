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
