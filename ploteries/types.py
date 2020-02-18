from sqlalchemy import types as sa_types
import json, plotly, plotly.graph_objects as go

class FigureType(sa_types.TypeDecorator):
    impl = sa_types.LargeBinary
    def process_bind_param(self, value, dialect):
        fig = value
        json_str = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        json_bytes = json_str.encode()
        return json_bytes
    def process_result_value(self, value, dialect):
        kwargs = json.loads(value.decode())
        return go.Figure(**kwargs)

class HistogramType(FigureType):
    pass
