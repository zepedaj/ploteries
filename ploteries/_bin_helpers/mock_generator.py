from .main import main, path_arg
import time
from tqdm import tqdm
import climax as clx
from time import sleep
from itertools import islice
import os.path as osp
import numpy as np
from ploteries.writer import Writer
from contextlib import nullcontext
from tempfile import TemporaryDirectory


class RandomWalk:
    def __init__(self, dim, mean=0, var=1.0, smoothing=0):
        self.dim = dim
        self.std = var**0.5
        self.curr = np.full(dim, mean)
        self.smoothing = float(min(max(0, smoothing), 1))

    def __iter__(self):
        return self

    def __next__(self):
        if (win_size := int(self.smoothing * self.dim)) > 1:
            curr = np.convolve(self.curr, (1.0 / win_size) * np.ones(win_size), "same")
        else:
            curr = self.curr
        self.curr = curr + np.random.randn(self.dim) * self.std
        return self.curr


@main.command()
@clx.option(
    "--out",
    default=None,
    help="Store the data store in this path (default uses a temporary path).",
)
@clx.option(
    "--interval",
    type=float,
    default=0.5,
    help="Number of seconds to wait before starting the next write operation.",
)
@clx.option(
    "--length",
    type=float,
    default=int(5e4),
    help="Number of timesteps - limits the total disk size of the generated data store.",
)
@clx.option(
    "--num-traces", type=int, default=3, help="Number of traces in each figure."
)
@clx.option(
    "--num-scalars", type=int, default=2, help="Number of `add_scalars` figures."
)
@clx.option("--num-plots", type=int, default=2, help="Number of `add_plots` figures.")
@clx.option(
    "--num-plot-points",
    type=int,
    default=50,
    help="Number of points in each `add_plots` figure.",
)
@clx.option(
    "--num-histograms", type=int, default=2, help="Number of `add_plots` figures."
)
def demo(
    out,
    interval,
    length,
    num_traces,
    num_scalars,
    num_plots,
    num_plot_points,
    num_histograms,
):
    """
    Creates a live mock data generator that can be used to showcase ploteries. To use it, first launch the mock generator, and then launch a ploteries server using the printed command:

    .. code-block:: bash

       # From shell 1
       $ ./ploteries demo
         <prints out a a full command with temporary path>

       # From shell 2, copy paste the command printed above
       $ ploteries launch --interval 2 <temporary path>
    """

    try:
        with nullcontext() if out else TemporaryDirectory() as root_dir:
            out = out or osp.join(root_dir, "data_store.pltr")
            #
            print(
                f"This command generates demo data. To visualize this data, you need to launch a ploteries server in a different shell using the following command:\n\tploteries launch --interval 2 --path {out}"
            )

            writer = Writer(out)
            try:
                # Define figures
                scalars = [
                    {"name": f"scalars/scalars-{_k}", "iter": RandomWalk(num_traces)}
                    for _k in range(num_scalars)
                ]

                plots = [
                    {
                        "name": f"plots/plots-{_k}",
                        "iter": [
                            RandomWalk(num_plot_points, var=0.01, smoothing=0.2)
                            for _ in range(num_traces)
                        ],
                    }
                    for _k in range(num_scalars)
                ]

                histograms = [
                    {
                        "name": f"histograms/histogram-{_k}",
                        "iter": [RandomWalk(500) for _ in range(num_traces)],
                    }
                    for _k in range(num_histograms)
                ]

                def table_data():
                    while True:
                        yield {f"Col {k}": np.random.randn() for k in range(5)}

                tables = [
                    {"name": "tables/table-1", "data": table_data(), "kwargs": {}},
                    {
                        "name": "tables/table-2",
                        "data": table_data(),
                        "kwargs": {"transposed": True},
                    },
                ]

                k = 0
                with tqdm(total=length) as pbar:
                    while True:
                        if k >= length:
                            break
                        k += 1

                        # Add scalars
                        for _scalar in scalars:
                            writer.add_scalars(
                                _scalar["name"], next(_scalar["iter"]), k
                            )

                        # Add plots
                        for _plot in plots:
                            writer.add_plots(
                                _plot["name"],
                                [{"y": next(_iter)} for _iter in _plot["iter"]],
                                k,
                            )

                        # Add histograms
                        for _histo in histograms:
                            writer.add_histograms(
                                _histo["name"],
                                [next(_iter) for _iter in _histo["iter"]],
                                k,
                            )

                        # Add tables
                        for _tbl in tables:
                            writer.add_table(
                                _tbl["name"], next(_tbl["data"]), k, **_tbl["kwargs"]
                            )

                        # Sleep
                        sleep(interval)
                        pbar.update(1)
            finally:
                # Wait for the writer thread to finish to avoid error messages.
                writer.flush()

            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:
        pass
