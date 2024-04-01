import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

checkpoint_dir = pathlib.Path('res') / 'checkpoint'
fontsize = 20

num_batch = 1_000_000

lr = 0.1
gamma = 0.99
batch_sizes = [1, 2, 4, 8]
seeds = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]

fig, ax = plt.subplots()
point_x = []
point_y = []
for batch_size in batch_sizes:
    value_errors = []
    for seed in seeds:
        if batch_size != 1:
            path = checkpoint_dir / str(seed) / f'B = {batch_size}, lr = {lr}, gamma = {gamma}' / f'value_error-batch_{num_batch}.npy'
        else:
            path = checkpoint_dir / str(seed) / f'B = {batch_size}, lr = {lr}, gamma = {gamma}' / f'value_error-batch_10000000.npy'

        value_errors.append(
            np.load(path)
        )

    xs = 1 / np.tile(batch_size, len(seeds))
    ys = [np.mean(x[900_000: 1_000_000]) for x in value_errors]
    sns.scatterplot(
        x=xs,
        y=ys,
        ax=ax,
        label=f'$B = {batch_size}$',
    )
    point_x = np.concatenate([point_x, xs])
    point_y = np.concatenate([point_y, ys])

sns.lineplot(
    x=np.unique(point_x),
    y=np.poly1d(np.polyfit(point_x, point_y, 1))(np.unique(point_x)),
    ax=ax,
    label='Linear fit',
)

ax.legend(fontsize=fontsize)
ax.set_xlabel('$1/B$', fontsize=fontsize)
ax.set_ylabel('Convergence Value Error', fontsize=fontsize)
fig.tight_layout()
fig.savefig(checkpoint_dir / 'mc-b-line.pdf', format='pdf', bbox_inches='tight')
