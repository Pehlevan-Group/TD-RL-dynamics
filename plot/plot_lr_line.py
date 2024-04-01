import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

checkpoint_dir = pathlib.Path('res') / 'checkpoint'
fontsize = 20

num_batch = 10_000_000

lrs = [0.01, 0.02, 0.05, 0.1, 0.2]
gamma = 0.99
seeds = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]

fig, ax = plt.subplots()
point_x = []
point_y = []
for lr in lrs:
    value_errors = []
    for seed in seeds:
        value_errors.append(
            np.load(checkpoint_dir / str(seed) / f'B = 1, lr = {lr}, gamma = {gamma}' / f'value_error-batch_{num_batch}.npy')
        )

    xs = np.tile(lr, len(seeds))
    ys = [np.mean(x[9_900_000: 10_000_000]) for x in value_errors]
    sns.scatterplot(
        x=np.tile(lr, len(seeds)),
        y=ys,
        ax=ax,
        label=f'$\eta = {lr}$',
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
ax.set_xlabel('Learning Rate', fontsize=fontsize)
ax.set_ylabel('Convergence Value Error', fontsize=fontsize)
fig.tight_layout()
fig.savefig(checkpoint_dir / 'mc-lr-line.pdf', format='pdf', bbox_inches='tight')
