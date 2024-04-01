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
for batch_size in batch_sizes:
    value_errors = []
    for seed in seeds:
        if batch_size != 1:
            path = checkpoint_dir / str(seed) / f'B = {batch_size}, lr = {lr}, gamma = {gamma}' / f'value_error-batch_{num_batch}.npy'
        else:
            path = checkpoint_dir / str(seed) / f'B = {batch_size}, lr = {lr}, gamma = {gamma}' / f'value_error-batch_10000000.npy'

        value_errors.append(
            np.load(path)[:num_batch]
        )


    def sample(arr, num_points_per_bin=100):
        # Sample logarithmically
        idx = np.linspace(0, int(np.log10(num_batch)), int(np.log10(num_batch) * num_points_per_bin + 1))
        idx = np.power(10, idx)
        # minus 1 for 0-based indexing
        idx = np.array(np.unique(np.floor(idx)), dtype=np.intp) - 1
        return arr[idx]


    xs = np.tile(sample(np.arange(num_batch)), len(seeds))
    ys = np.ravel([sample(value_error) for value_error in value_errors])

    sns.lineplot(
        x=xs,
        y=ys,
        ax=ax,
        errorbar=('ci', 95),
        label=f'$B = {batch_size}$',
    )

ax.set(xscale="log", yscale="log")
ax.set_ylim([10, 10_000_000])
ax.set_xlim([1, 1_000_000])
ax.legend(fontsize=fontsize)
ax.set_xlabel('Batch', fontsize=fontsize)
ax.set_ylabel('Value Error', fontsize=fontsize)
fig.tight_layout()
fig.savefig(checkpoint_dir / 'mc-b-value_error-low_res.pdf', format='pdf', bbox_inches='tight')
