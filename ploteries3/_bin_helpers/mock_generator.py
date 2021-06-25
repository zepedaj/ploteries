from .main import main, path_arg
from tqdm import tqdm
import climax as clx
from time import sleep
from itertools import islice
import os.path as osp
import numpy as np
from ploteries3.writer import Writer
from contextlib import nullcontext
from tempfile import TemporaryDirectory


@main.command()
@clx.option('--out', default=None,
            help='Store the data store in this path (default uses a temporary path).')
@clx.option('--interval', type=float, default=0.5,
            help='Number of seconds to wait before starting the next write operation.')
@clx.option('--length', type=float, default=int(1e4),
            help='Number of timesteps - limits the total disk size of the generated data store.')
def launch_mock_generator(out, interval, length):
    """
    Creates a live mock data generator that can be used to showcase ploteries. To use it, first launch the mock generator, and then launch a ploteries server using the printed command:

    .. code-block:: bash

       # From shell 1
       $ ./ploteries launch_mock_generator
         Launch a ploteries server with the following command:
             ploteries launch --interval 2 /tmp/tmp_55f_7jh/data_store.pltr
         0%|                             | 0/10000 [00:00<?, ?it/s]

       # From shell 2
       $ ploteries launch --interval 2 /tmp/tmp_55f_7jh/data_store.pltr
    """

    try:
        with (nullcontext() if out else TemporaryDirectory()) as root_dir:
            out = out or osp.join(root_dir, 'data_store.pltr')
            #
            print(
                f'Launch a ploteries server with the following command (ctrl+c to exit):\n\tploteries launch --interval 2 {out}')

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

            writer = Writer(out)
            k = -1
            N = 3
            scalars1 = iter(random_walks(N=N))
            scalars2 = iter(random_walks(N=N))
            plot1 = iter(random_walk())
            plot2 = iter(random_walk())
            histo1 = iter(random_walk())
            histo2 = iter(random_walk())

            with tqdm() as pbar:
                while True:
                    k += 1
                    # Add scalars
                    writer.add_scalars('scalars/scalars1', next(scalars1), k,
                                       [{'name': f'plot {_l}'} for _l in range(N)])
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
                    pbar.update(1)

    except KeyboardInterrupt:
        pass
