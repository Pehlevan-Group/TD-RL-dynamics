import argparse
import math
import pathlib

import gym
import joblib
import numpy as np
from tqdm import tqdm


class MountainCar:
    def __init__(self, num_states, max_steps, q=None):
        self.env = gym.make('MountainCar-v0')

        self.num_states = num_states
        self.max_steps = max_steps

        if q is None:
            # Load Q table
            self.q = np.load('mountain_car_q.npy')
        else:
            self.q = q

        # Precompute the optimal policy
        self.policy = np.argmax(self.q, axis=-1)

    def discretize_state(self, state):
        """
        Discretize the state into a tuple of integers

        :param state: numpy array of shape (N,)
        :return: tuple of integers that represents the state
        """
        spacings = (self.env.observation_space.high - self.env.observation_space.low) / self.num_states
        ret = np.floor((state - self.env.observation_space.low) / spacings)

        # Clip to prevent floating point errors
        ret = np.clip(ret, 0, self.num_states - 1).astype(int)
        return ret

    def get_action(self, s_discrete):
        """
        Obtain an action given the state.

        :param s_discrete: Discretized state.
        :return:
        """
        return self.policy[tuple(s_discrete)]

    def sample_episodes(self, num_episodes, episode_length, seed, log_every=100):
        env = gym.make('MountainCar-v0')

        all_states = np.zeros((num_episodes, episode_length + 1, len(self.num_states)), dtype=np.int8)
        all_rews = np.zeros((num_episodes, episode_length), dtype=np.int8)

        with tqdm(total=num_episodes, desc='Sampling episodes') as pbar:
            for episode_i in range(num_episodes):
                # Initialize the state
                if episode_i == 0:
                    s, _ = env.reset(seed=seed)
                else:
                    s, _ = env.reset()

                # Episode states and rewards
                states = []  # size episode_length + 1 at the end
                rews = []  # size episode_length

                state = self.discretize_state(s)

                # Sample an episode
                states.append(state)  # size episode_length + 1 at the end
                for _ in range(episode_length):
                    # Take the action
                    a = self.get_action(state)
                    s_next, rew, done, _, _ = env.step(a)
                    next_state = self.discretize_state(s_next)
                    states.append(next_state)
                    rews.append(rew)
                    state = next_state

                    if done:
                        break

                # Extend episode
                if len(states) != episode_length + 1:
                    states += [states[-1]] * (episode_length + 1 - len(states))
                    rews += [0] * (episode_length - len(rews))

                all_states[episode_i] = states
                all_rews[episode_i] = rews

                if (episode_i + 1) % log_every == 0:
                    pbar.update(log_every)

        return all_states, all_rews

    def sample_all_batches(self, episode_length, num_episodes, num_envs, save_every=None, seed_offset=0, res_dir=None):
        """
        Sample all batches of episodes.

        :param episode_length: The length of each episode.
        :param num_episodes: The number of episodes to sample.
        :param num_envs: The number of environments to use in parallel.
        :param save_every: If not None, save every save_every episodes.
        :param seed_offset: The offset to use for the seed.
        :param res_dir: The directory to save the results to.
        """

        # Create res_dir if it does not exist
        if res_dir is not None:
            pathlib.Path(res_dir).mkdir(parents=True, exist_ok=True)
        else:
            res_dir = pathlib.Path('./')

        # Sample all batches
        batch_states = np.zeros((math.ceil(num_episodes / num_envs) * num_envs, episode_length + 1, len(self.num_states)), dtype=np.int8)
        batch_rews = np.zeros((math.ceil(num_episodes / num_envs) * num_envs, episode_length), dtype=np.int8)

        num_episodes_per_env = math.ceil(num_episodes / num_envs)
        ret = joblib.Parallel(n_jobs=num_envs)(joblib.delayed(self.sample_episodes)(num_episodes_per_env, episode_length, seed) for seed in range(seed_offset, seed_offset + num_envs))
        for i in range(num_envs):
            # Extend list of batches
            batch_states[i * num_episodes_per_env: (i + 1) * num_episodes_per_env] = ret[i][0]
            batch_rews[i * num_episodes_per_env: (i + 1) * num_episodes_per_env] = ret[i][1]

            # TODO: Implement save_every
            # if save_every is not None and i % save_every == 0:
            #     np.save(res_dir / 'batch_states.npy', batch_states.astype(np.int8))
            #     np.save(res_dir / 'batch_rews.npy', batch_rews.astype(np.int8))

        np.save(res_dir / 'batch_states.npy', batch_states.astype(np.int8))
        np.save(res_dir / 'batch_rews.npy', batch_rews.astype(np.int8))


def main():
    # Setup arguments
    parser = argparse.ArgumentParser(
        description="""
Obtain samples in parallel for value evaluation.
    Example usage:
        python -m simulation.mountain_car_get_samples --num_envs 8 --num_episodes 1000 --save_every 100 --seed_offset 0
    """,
    )
    parser.add_argument('--num_envs', type=int, default=2, help='Number of environments to use in parallel.')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes to sample.')
    parser.add_argument('--save_every', type=int, default=None, help='If not None, save every save_every episodes.'
                                                                     'NOTE: To be implemented.')
    parser.add_argument('--seed_offset', type=int, default=0, help='Offset of the seed to use for the environments.'
                                                                   'The seed used will be seed_offset * num_envs'
                                                                   '+ [0, num_envs].')
    args = parser.parse_args()

    res_dir = pathlib.Path('res')
    num_states = np.array([42, 28])
    episode_length = 350

    # Get the q-learning results
    q = np.load(res_dir / 'mountain_car_q' / '42_28.npy')

    # Create the environment
    agent = MountainCar(num_states, episode_length, q=q)

    # Obtain samples
    agent.sample_all_batches(
        episode_length,
        args.num_episodes,
        args.num_envs,
        save_every=args.save_every,
        seed_offset=args.seed_offset,
        res_dir=res_dir / 'samples' / f'{args.seed_offset}',
    )


if __name__ == '__main__':
    main()

# Code for combining states and rews
# batch_states = np.load('batch_states.npy')
# batch_states_extended = np.load('batch_states-extended.npy')
# np.save('batch_states_all.npy', np.concatenate([batch_states, batch_states_extended]))
# del batch_states, batch_states_extended
# batch_rews = np.load('batch_rews.npy')
# batch_rews_extended = np.load('batch_rews-extended.npy')
# np.save('batch_rews_all.npy', np.concatenate([batch_rews, batch_rews_extended]))
# del batch_rews, batch_rews_extended
