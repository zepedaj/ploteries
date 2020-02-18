from sqlalchemy import create_engine, MetaData, types as sa_types, select
from . import types as ps_types
from .writer import sqlite3_concurrent_engine

class Reader(object):
    #
    def __init__(self, path):
        #
        self.path=path
        #
        #creator = lambda: sqlite3.connect('file:' + self.path + '?mode=ro', uri=True)
        self._engine = sqlite3_concurrent_engine(path)
        self._connection = self._engine.connect()
        self._metadata = MetaData(bind=self._engine)        
        self.reflect()

    def _execute(self, cmd): return self._connection.execute(cmd)
    def reflect(self):

        #
        self._metadata.reflect()
        self._table_meta = [dict(_x) for _x in self._execute(select([self._metadata.tables['__ps__table_meta']]))]

        # Cast derived types
        for _curr_meta in self._table_meta:
            table=self._metadata.tables[_curr_meta['name']]
            type_class = None
            try:
                type_class = getattr(ps_types, _curr_meta['content_type'])
            except AttributeError: pass
            if type_class is not None:
                table.columns['content'].type=type_class()

    def tables_of_type(self, content_type, key=lambda _e:_e['name']):
        #return {key(_e):self._metadata.tables[_e['name']] for _e in self._table_meta if 
        #        _e['content_type']==content_type}
        return [_e for _e in self._table_meta if _e['content_type']==content_type]
    
    def select(self, table_name, global_step=None):
        table = self._metadata.tables[table_name]
        #
        sel = select([table.c.global_step,table.c.content])
        if global_step is not None:
            sel = sel.where(table.c.global_step == global_step) #CB2: Choose most recent if more than 1.
        else:
            sel = sel.order_by(table.c.global_step) #CB2: Ensure at most one record per global step.
        #
        return self._execute(sel)

    def global_steps(self, table_name):
        table = self._metadata.tables[table_name]
        result = self._execute(select([table.c.global_step]).order_by('global_step').distinct())
        return [_x['global_step'] for _x in result]
