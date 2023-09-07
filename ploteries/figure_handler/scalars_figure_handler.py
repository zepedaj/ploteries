from copy import deepcopy
from typing import Tuple

from plotly.colors import convert_colors_to_same_type, unlabel_rgb
from .figure_handler import FigureHandler
import numpy as np
import plotly.express as px
import colorsys


class ScalarsFigureHandler(FigureHandler):
    smoothing_n = 100

    def build_figure(self, *args, **kwargs):
        increase_lightness = 0.5
        fig = super().build_figure(*args, **kwargs)

        default_colors = iter(Colors())
        default_light_colors = iter(Colors(increase_lightness=increase_lightness))

        for scatter_plot in fig.data:
            if scatter_plot.line.color:
                base_color = unlabel_rgb(scatter_plot.line.color)
                scatter_color = Colors.rgb_to_hsl(
                    base_color,
                    lightness_multiplier=increase_lightness,
                    normalized=False,
                )
                smoothed_scatter_color = Colors.rgb_to_hsl(base_color, normalized=False)
            else:
                scatter_color = next(default_light_colors)
                smoothed_scatter_color = next(default_colors)

            scatter_plot.line.color = scatter_color

            smoothed_scatter_plot = deepcopy(scatter_plot)
            smoothed_scatter_plot.line.color = smoothed_scatter_color
            smoothed_scatter_plot.y = self.smoothen(scatter_plot.y)
            scatter_plot["showlegend"] = False
            if scatter_plot.name is not None:
                smoothed_scatter_plot.name = scatter_plot.name
                scatter_plot.name = scatter_plot.name + "_unsmoothed"
            else:
                smoothed_scatter_plot["showlegend"] = False
            fig.add_trace(smoothed_scatter_plot)

        return fig

    def smoothen(self, x):  # TODO: Should pass this to javascript.
        smoothing_n = self.smoothing_n
        if x is None:
            return x

        def smoothing_kernel(smoothing_n):
            # The nth point will contribute 1e-2 as much as the first point.
            w = (
                np.array([1.0])
                if smoothing_n == 1
                else np.exp(np.arange(smoothing_n) * (np.log(1e-2) / (smoothing_n - 1)))
            )
            return w / w.sum()

        w = smoothing_kernel(smoothing_n)
        smoothed = np.convolve(x, w, mode="full")[: len(x)]
        # Normalize the filter in the first points.
        smoothed[:smoothing_n] /= (w.cumsum())[: len(smoothed)]
        return smoothed


class Colors:
    def __init__(self, name="Plotly", increase_lightness=0):
        """
        scale_lightness: [0,1]
        """
        self._rgb = list(
            map(
                lambda rgb: (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255),
                map(px.colors.hex_to_rgb, getattr(px.colors.qualitative, name)),
            )
        )
        self.increase_lightness = increase_lightness

    def __iter__(self):
        k = 0
        while True:
            yield self[k]
            k += 1

    def __len__(self):
        return len(self._rgb)

    @staticmethod
    def rgb_to_hsl(rgb: Tuple[float], lightness_multiplier=0.0, normalized=True):
        if not normalized:
            rgb = tuple([x / 255 for x in rgb])
        hls = colorsys.rgb_to_hls(*rgb)
        hsl = hls[0], hls[2], hls[1] + (1.0 - hls[1]) * lightness_multiplier
        out = f"hsl({360*hsl[0]:.0f}, {hsl[1]:.0%}, {hsl[2]:.0%})"
        return out

    def __getitem__(self, k):
        k = k % len(self._rgb)
        out = self.rgb_to_hsl(self._rgb[k], self.increase_lightness)
        return out
