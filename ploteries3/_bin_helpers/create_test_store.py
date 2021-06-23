from .main import main, path_arg
import climax as clx
from time import sleep
from itertools import islice
import os.path as osp
import numpy as np
from ploteries3 import writer_api


@main.command(parents=[path_arg])
@clx.option('--overwrite', action='store_true', help='overwrite database', default=False)
@clx.option('--interval', type=float,
            help='Number of seconds to wait before starting the next write operation.')
def create_test_store(path, overwrite, interval):
    """
    Create a test database.
    """
    if overwrite:
        with open(path, 'w'):
            pass
    elif osp.isfile(path):
        raise Exception('File already exists.')

    # Data generation
    def random_walk(start=0, var=1.0):
        val, std = start, var**0.5
        while True:
            yield val + np.random.randn()*(std/10)
            val += np.random.randn()*std

    def random_walks(N=3, start=0, var=1.0):
        walkers = [iter(random_walk(start=start, var=var)) for _ in range(N)]
        while True:
            yield [next(_w) for _w in walkers]

    writer = writer_api.Writer(path)
    k = -1
    N = 3
    scalars1 = iter(random_walks(N=N))
    scalars2 = iter(random_walks(N=N))
    plot1 = iter(random_walk())
    plot2 = iter(random_walk())
    histo1 = iter(random_walk())
    histo2 = iter(random_walk())
    while True:
        k += 1
        print(k)
        # Add scalars
        writer.add_scalars('scalars/scalars1', next(scalars1), k,
                           names=[f'plot{_l}' for _l in range(N)])
        writer.add_scalars('scalars/scalars2', next(scalars2), k)

        # # Add plots
        # X = list(range(50))
        # if k % 100 == 0:
        #     writer.add_plots(
        #         'plots/plot1', [{'x': X, 'y': list(islice(plot1, 50))} for _ in range(N)],
        #         k, names=[f'plot{_l}' for _l in range(N)])
        #     writer.add_plots(
        #         'plots/plot2', [{'x': X, 'y': list(islice(plot2, 50))} for _ in range(N)],
        #         k, names=[f'plot{_l}' for _l in range(N)])

        # # Add histograms
        # if k % 100 == 0:
        #     writer.add_histograms('histograms/histogram1',
        #                           [list(islice(histo1, 1000)) for _ in range(N)],
        #                           k, names=[f'histo{_l}' for _l in range(N)])
        #     writer.add_histograms('histograms/histogram2',
        #                           [list(islice(histo2, 1000)) for _ in range(N)],
        #                           k, names=[f'histo{_l}' for _l in range(N)])

        # Sleep
        sleep(interval)
