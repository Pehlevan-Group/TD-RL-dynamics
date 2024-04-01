import argparse
import itertools
import logging
import pathlib

import jax
import jax.numpy as jnp
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm


class MountainCar:
    def __init__(self, num_states):
        self.num_states = num_states

    def sim_value_error_parallelized(
            self,
            feature_size,
            gamma,
            feature_func,
            episode_length,
            start_with_w=None,
            true_value=None,
            checkpoint_dir=None,
            checkpoint_every_n_batch=100,
            learning_rate=0.01,
            num_episode_per_batch=1,
            num_batch=1000,
            sample_path=None,
        ):
        """ Simulate the value error of a linear value function approximation.

        States start from 0.

        :param feature_size: NumPy array. The shape of the feature vector.
        :param gamma: Float. The discount factor.
        :param feature_func: (Integer) -> (num_feature, ) NumPy array. The feature function.
        :param episode_length: Integer. The length of each episode.
        :param start_with_w: (feature_size, ) NumPy array. The initial weights.
        :param true_value: (num_state, ) NumPy array. The true value function.
        :param checkpoint_dir: String. The directory to save the checkpoints.
        :param checkpoint_every_n_batch: Integer. Save the checkpoint every n batches.
        :param learning_rate: Float. The learning rate.
        :param num_episode_per_batch: Integer. The number of episodes to run before updating the weights.
        :param num_batch: Integer. The number of batches to run.
        :param sample_path: Path. The path to the directory of the samples. The directory should contain
            batch_states.npy and batch_rews.npy.
        :return:
        """

        if true_value is None:
            true_value = np.zeros(self.num_states)
        true_value_jax = jax.device_put(true_value)

        # Obtain all features
        features = np.zeros(np.concatenate((self.num_states, feature_size)))  # (num_states, feature_size)
        for s in itertools.product(*[range(x) for x in self.num_states]):
            features[s] = feature_func(s)
        features_jax = jax.device_put(features)

        # Initialize the value function errors
        value_error = np.zeros((num_batch, ))
        v_hat_norm = np.zeros((num_batch, ))
        max_err = np.zeros((num_batch, ))

        # Obtain all batches
        logging.info("Obtaining all batches...")
        if sample_path is not None and (sample_path / 'batch_states.npy').exists() and (sample_path / 'batch_rews.npy').exists():
            batch_states = np.load(sample_path / 'batch_states.npy')
            batch_rews = np.load(sample_path / 'batch_rews.npy')
        else:
            # Should always get samples first for efficiency
            raise NotImplementedError

        if len(batch_states) < num_batch * num_episode_per_batch:
            raise ValueError("Not enough samples")

        def td_update(states, rews, w, v_hat=None):
            # Estimated value function
            # Sum the last len(feature_size) dimensions
            if v_hat is None:
                v_hat = jnp.einsum('ijklm, klm -> ij', features_jax, w)  # (*num_states, )

            # Select features for each state
            features_selected = jnp.array([features_jax[tuple(episode.T)] for episode in states[:, :-1]])

            # Compute deltas, note there is no reward for the last state
            v_hats = jnp.array([v_hat[tuple(episode.T)] for episode in states])
            deltas = rews + gamma * v_hats[:, 1:] - v_hats[:, :-1]
            w_new = w + learning_rate * jnp.einsum('ijklm, ij -> klm', features_selected,
                                                   deltas) / num_episode_per_batch / episode_length
            v_hat_new = jnp.einsum('ijklm, klm -> ij', features_jax, w_new)  # (*num_states, )

            # Weight the value error by the number of times we visited each state
            value_error_batch = jnp.sum((v_hat - true_value_jax) ** 2)
            v_hat_norm_batch = jnp.sum(v_hat ** 2)
            max_err_batch = jnp.max(jnp.abs(v_hat_new - v_hat))

            return w_new, v_hat_new, value_error_batch, v_hat_norm_batch, max_err_batch

        td_update_jit = jax.jit(td_update)

        # Initialize the weights
        if start_with_w is None:
            w = np.zeros(feature_size)
        else:
            assert(start_with_w.shape == feature_size)
            w = start_with_w
        v_hat = None

        logging.info("Performing TD updates...")
        with tqdm(total=num_batch) as pbar:
            for batch in range(num_batch):
                # Get batch (num_episode_per_batch, episode_length)
                states = batch_states[batch * num_episode_per_batch: (batch + 1) * num_episode_per_batch]
                rews = batch_rews[batch * num_episode_per_batch: (batch + 1) * num_episode_per_batch]
                states = jnp.array(states)
                rews = jnp.array(rews)

                # Perform TD update
                w, v_hat, value_error[batch], v_hat_norm[batch], max_err[batch] = td_update_jit(
                    states,
                    rews,
                    w,
                    v_hat=v_hat
                )

                # Save progress after every checkpoint_every_n_batch
                if checkpoint_dir is not None and batch % checkpoint_every_n_batch == 0:
                    np.save(checkpoint_dir / f'w-batch_{batch}.npy', w)
                    np.save(checkpoint_dir / f'v_hat-batch_{batch}.npy', v_hat)
                    np.save(checkpoint_dir / f'value_error-batch_{num_batch}.npy', value_error)

                if (batch + 1) % checkpoint_every_n_batch == 0:
                    pbar.update(checkpoint_every_n_batch)

        np.save(checkpoint_dir / f'max_err-batch_{num_batch}.npy', max_err)
        np.save(checkpoint_dir / f'v_hat-batch_{num_batch}.npy', v_hat)
        np.save(checkpoint_dir / f'v_hat_norm-batch_{num_batch}.npy', v_hat_norm)
        np.save(checkpoint_dir / f'w-batch_{num_batch}.npy', w)

def main():
    # Setup arguments
    parser = argparse.ArgumentParser(
        description="""
Use samples to perform policy evaluation.
    Example usage:
        python -m simulation.mountain_car --seed_offset 0 --num_episode_per_batch 1 --num_episode_per_batch_true_value 1 --num_batch 1000 --num_batch_true_value 1000
        """,
    )
    parser.add_argument('--seed_offset', type=int, default=0,
                        help='Seed offset used during sampling. This only affects the samples used.')
    parser.add_argument('--num_episode_per_batch', type=str, default='1',
                        help='Number of episodes per batch for the student as a comma separated string.')
    parser.add_argument('--num_episode_per_batch_true_value', type=int, default=1,
                        help='Number of episodes per batch for the teacher.')
    parser.add_argument('--num_batch', type=int, default=1000,
                        help='Number of batches used to train the student.')
    parser.add_argument('--num_batch_true_value', type=int, default=1000,
                        help='Number of batches used to train the teacher.')
    parser.add_argument('--lrs', type=str, default='0.1',
                        help='Learning rates to use as a comma separated string. If not provided, will default to 0.1.')
    parser.add_argument('--learning_rate_true_value', type=float, default=0.01,
                        help='Learning rate used to train the teacher.')
    parser.add_argument('--gamma', type=str, default='0.99',
                        help='Discount factors separated by commas. If not provided, will default to 0.99.')

    parser.add_argument('--sample_path', type=str, default=None,
                        help='Path to the samples. If not provided, will default to '
                             'res / \'samples\'.')
    parser.add_argument('--skip_train_true_value', action='store_true',
                        help='If provided, will skip training the teacher.')
    parser.add_argument('--num_envs', type=int, default=1,
                        help='Number of environments to use in parallel.')
    args = parser.parse_args()

    res_dir = pathlib.Path('res')
    checkpoint_dir = res_dir / 'checkpoint' / str(args.seed_offset)
    checkpoint_every_n_batch = 1000
    num_states = np.array([42, 28])
    feature_size = np.array([42, 28])
    learning_rate_true_value = args.learning_rate_true_value
    episode_length = 350
    num_episode_per_batch = args.num_episode_per_batch
    num_episode_per_batch_true_value = args.num_episode_per_batch_true_value
    num_batch = args.num_batch
    num_batch_true_value = args.num_batch_true_value
    skip_train_true_value = args.skip_train_true_value
    if args.sample_path is None:
        sample_path = res_dir / 'samples'
    else:
        sample_path = pathlib.Path(args.sample_path)
    sample_path = sample_path / str(args.seed_offset)

    # Create the environment
    agent = MountainCar(num_states)

    def get_feature_func(feature_type, options={}):
        def feature_func(s):
            if feature_type == 'fourier':
                # We need both cosines and sines to represent any function
                grid = np.meshgrid(*[range(x) for x in feature_size])
                grid = [g.T for g in grid]
                grid = np.stack(grid, axis=-1)
                grid = np.dot(2 * np.pi * grid / feature_size, s)
                features = np.stack([np.cos(grid), np.sin(grid)], axis=-1)
                # Final answer has a shape of (*feature_size, 2)
                return features

            elif feature_type == 'weighted_fourier':
                # We need both cosines and sines to represent any function
                features = np.zeros(np.concatenate([feature_size, [2, ]]))
                for p in itertools.product(*[range(x) for x in feature_size]):
                    features[np.concatenate([p, [0, ]])] = np.cos(np.dot(2 * np.pi * p / feature_size, s))
                    features[np.concatenate([p, [1, ]])] = np.sin(np.dot(2 * np.pi * p / feature_size, s))
                return features

            # elif feature_type == 'grid':
            #     # Grid cells as a basis
            #     if 'sigma' not in options:
            #         options['sigma'] = 0.5
            #
            #     def gaussian(x, mean, sigma):
            #         return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mean) ** 2 / (2 * sigma ** 2))
            #
            #     # Lattice points are every feature-th state
            #     ret = []
            #     for feature in range(1, num_feature + 1):
            #         lattice_points = list(range(0, num_state, feature))
            #         ret.append(np.sum([gaussian(s, lattice_point, options['sigma']) for lattice_point in lattice_points]))
            #     return np.array(ret)
            else:
                return np.array([s])

        return feature_func

    # Plot value errors for different learning rates
    lrs = [float(lr) for lr in args.lrs.split(',')]
    batch_sizes = [int(batch_size) for batch_size in args.num_episode_per_batch.split(',')]
    gammas = [float(gamma) for gamma in args.gamma.split(',')]

    def get_value_error(lr, batch_size, gamma):
        # Get the true value function
        save_dir_true = res_dir / 'true_v_td' / str(
            args.seed_offset) / f'B = {num_episode_per_batch_true_value}, lr = {learning_rate_true_value}, gamma = {gamma}'
        if not save_dir_true.exists():
            save_dir_true.mkdir(parents=True)

        if not skip_train_true_value:
            agent.sim_value_error_parallelized(
                np.concatenate([feature_size, [2, ]]),
                gamma,
                get_feature_func('fourier'),
                episode_length,
                checkpoint_dir=save_dir_true,
                checkpoint_every_n_batch=checkpoint_every_n_batch,
                learning_rate=learning_rate_true_value,
                num_episode_per_batch=num_episode_per_batch_true_value,
                num_batch=num_batch_true_value,
                sample_path=sample_path,
            )

        true_value = np.load(save_dir_true / f'v_hat-batch_{num_batch_true_value}.npy')

        # Construct save directory
        save_dir = checkpoint_dir / f'B = {batch_size}, lr = {lr}, gamma = {gamma}'
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        agent.sim_value_error_parallelized(
            np.concatenate([feature_size, [2, ]]),
            gamma,
            get_feature_func('fourier'),
            episode_length,
            checkpoint_dir=save_dir,
            checkpoint_every_n_batch=checkpoint_every_n_batch,
            true_value=true_value,
            learning_rate=lr,
            num_episode_per_batch=batch_size,
            num_batch=num_batch,
            sample_path=sample_path,
        )

    joblib.Parallel(n_jobs=args.num_envs)(joblib.delayed(get_value_error)(lr, b, g) for lr in lrs for b in batch_sizes for g in gammas)

    # fig, ax = plt.subplots()
    # for lr in lrs:
    #     value_error = np.load(checkpoint_dir / f'B = {num_episode_per_batch}, lr = {lr}' / f'value_error-batch_{num_batch}.npy')
    #     sns.lineplot(
    #         x=np.tile(np.arange(num_batch), num_trials),
    #         y=np.ravel(value_error),
    #         ax=ax,
    #         errorbar=('ci', 95),
    #         label=f'lr = {lr}'
    #     )
    # ax.set(xscale="log", yscale="log")
    # ax.set_ylim([0.1, 10_000_000])
    # ax.set_xlim([0.1, 10_000_000])
    # ax.set_xlabel('Batch')
    # ax.set_ylabel('Value function error')
    # fig.savefig(checkpoint_dir / f'value_error_lr-batch_{num_batch}.png')


if __name__ == '__main__':
    main()
