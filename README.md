# TD-RL-dynamics

Theory of Temporal Difference Learning Dynamics for High Dimensional Features from our recent [preprint](https://arxiv.org/abs/2307.04841).

The notebook contains code to reproduce the experimental tests of the theory from the paper. 

## How to Run MountainCar-v0 Simulations

1. Install required packages.
 
```commandline
# Create a new conda environment (if needed)
mamba create --name rl_learning_curves python=3.10
mamba activate rl_learning_curves

# Required packages
mamba install tqdm seaborn joblib gym=0.26.1 -y
# OR
# pip install gym==0.26.1
mamba install jaxlib=*=*cuda* jax=0.4.21 -y
```

2. Get samples using `mountain_car_get_samples`

Since it takes a lot of time to sample the policy, we sample the environment
in parallel first and then run the TD algorithm.

Variables:

```python
episode_length = 350  # steps for each episode
num_episode_per_batch = 1  # how many episodes per batch
num_batch = 1_000_000  # how many batches in total
num_envs = 4  # how many environments to sample in parallel
save_every = 50_000  # save to disk every num_batch * num_episode_per_batch // num_envs // save_every episodes
seed_offset = 0  # the seed for the first environment (therefore the program will use seed_offset + 1, seed_offset + 2, 
                 # ... for the other environments)
```

You can specify the variables using command line arguments. For example:

```commandline
python -m simulation.mountain_car_get_samples \
    --num_envs 32 \
    --num_episodes 10000000 \
    --seed_offset 0
```

3. Run policy evaluation using the sampled episodes. You can specify the command line arguments as follows.

### Compare between batch sizes

```commandline
XLA_PYTHON_CLIENT_MEM_FRACTION=.25 python -m simulation.mountain_car_jax \
    --seed_offset 0 \
    --num_episode_per_batch 1,2,4,8 \
    --num_episode_per_batch_true_value 1 \
    --num_batch 1000000 \
    --num_batch_true_value 10000000 \
    --num_envs 4 \
    --lrs 0.1 \
    --sample_path res/samples
```

### Compare between learning rates

```commandline
XLA_PYTHON_CLIENT_MEM_FRACTION=.15 python -m simulation.mountain_car_jax \
    --seed_offset 0 \
    --num_episode_per_batch 1 \
    --num_episode_per_batch_true_value 1 \
    --num_batch 1000000 \
    --num_batch_true_value 10000000 \
    --num_envs 5 \
    --lrs 0.01,0.02,0.05,0.1,0.2 \
    --skip_train_true_value
```

4. Plot the results

```commandline
python plot/plot_batch_line.py
python plot/plot_batch_loss.py
python plot/plot_lr_line.py
python plot/plot_lr_loss.py
```

