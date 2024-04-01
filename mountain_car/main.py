import gym
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import seaborn as sns
from tqdm import tqdm


def discretize_state(state, low_ob_space, high_ob_space, num_states):
    """
    Discretize the state into a tuple of integers

    :param state: numpy array of shape (N,)
    :param low_ob_space: numpy array of shape (N,)
    :param high_ob_space: numpy array of shape (N,)
    :param num_states: numpy array of shape (N,) where each item is the number of states in that dimension
    :return: tuple of integers that represents the state
    """
    spacings = (high_ob_space - low_ob_space) / num_states
    ret = np.floor((state - low_ob_space) / spacings)

    # Clip to prevent floating point errors
    ret = np.clip(ret, 0, num_states - 1).astype(int)
    return ret


def train(num_states):
    # Parameters
    num_trials = 1
    num_episodes = 2000
    max_steps = 1000
    learning_rate = 0.01
    gamma = 0.99
    epsilon = 0.1

    # Setup environment
    env = gym.make('MountainCar-v0')

    # Measure statistics
    episode_rewards = np.zeros((num_trials, num_episodes))

    for trial in range(num_trials):
        # Initialize the random number generator
        rng = default_rng(trial)

        # Initialize Q table
        q = rng.random(np.concatenate((num_states, [env.action_space.n, ])))

        for episode in tqdm(range(num_episodes)):
            s, _ = env.reset()
            s_discrete = discretize_state(s, env.observation_space.low, env.observation_space.high, num_states)
            episode_reward = 0

            for step in range(max_steps):
                # Choose action
                if rng.random() < epsilon:
                    a = env.action_space.sample()
                else:
                    a = np.argmax(q[tuple(s_discrete)])

                # Take action
                s_prime, r, done, _, _ = env.step(a)
                s_prime_discretized = discretize_state(s_prime, env.observation_space.low, env.observation_space.high, num_states)
                episode_reward += r

                # Update Q table
                s_a = tuple(np.concatenate((s_discrete, [a, ])))
                s_prime_a_prime = tuple(s_prime_discretized)
                q[s_a] += learning_rate * (r + gamma * np.max(q[s_prime_a_prime]) - q[s_a])

                # Update state
                s_discrete = s_prime_discretized

                if done:
                    break

            episode_rewards[(trial, episode)] = episode_reward

    # Plot rewards
    fig, ax = plt.subplots()
    sns.lineplot(
        x=np.tile(np.arange(num_episodes), num_trials),
        y=np.ravel(episode_rewards),
        ax=ax,
        errorbar=('ci', 95),
        label='Episode reward',
    )
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode reward')
    fig.savefig('res/mountain_car_q/learning_curve-42_28.png')

    # Save Q table
    np.save('res/mountain_car_q/42_28.npy', q)


def test(num_states, num_episodes=1000, max_steps=1000):
    # Setup environment
    env = gym.make('MountainCar-v0')

    # Load Q table
    q = np.load('res/mountain_car_q/42_28.npy')

    # Test the policy for how many steps it takes to reach the goal
    steps = np.zeros(num_episodes)
    for episode in tqdm(range(num_episodes)):
        s, _ = env.reset()
        s_discrete = discretize_state(s, env.observation_space.low, env.observation_space.high, num_states)

        for step in range(max_steps):
            a = np.argmax(q[tuple(s_discrete)])
            s, _, done, _, _ = env.step(a)
            s_discrete = discretize_state(s, env.observation_space.low, env.observation_space.high, num_states)
            if done:
                steps[episode] = step
                break

    print(f'Mean : {np.mean(steps)}')
    print(f'Std  : {np.std(steps)}')
    print(f'Min  : {np.min(steps)}')
    print(f'Max  : {np.max(steps)}')


def render(num_states, max_steps=1000):
    # Setup environment
    env = gym.make('MountainCar-v0', render_mode='human')

    # Load Q table
    q = np.load('res/mountain_car_q/42_28.npy')

    # Render the final policy
    while True:
        s, _ = env.reset()
        s_discrete = discretize_state(s, env.observation_space.low, env.observation_space.high, num_states)

        for step in range(max_steps):
            env.render()
            a = np.argmax(q[tuple(s_discrete)])
            s, _, done, _, _ = env.step(a)
            s_discrete = discretize_state(s, env.observation_space.low, env.observation_space.high, num_states)
            if done:
                break


def main():
    num_states = np.array([42, 28])

    skip_train = False
    skip_render = True
    if not skip_train:
        train(num_states)
    test(num_states)
    if not skip_render:
        render(num_states)


if __name__ == '__main__':
    main()
