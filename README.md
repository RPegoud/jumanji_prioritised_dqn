# Solving Jumanji Environments with Prioritised DQN

| Environment                                                                     | Timesteps | Time    | Time per 100,000 steps | Converged | Uses CNN |
|:------------------------------------------------------------------------------- |:--------- | ------- | ---------------------- | --------- |:-------- |
| [``CVRP-v1``](https://instadeepai.github.io/jumanji/environments/cvrp/)         | 1.200.000 | 7min12s | 36s                    | No        | No       |
| [``Knapsack-v1``](https://instadeepai.github.io/jumanji/environments/knapsack/) | 1.200.000 | 8min20s | 43s                    | Yes       | No       |
| [``Maze-v0``](https://instadeepai.github.io/jumanji/environments/maze/)         | 1.200.000 | 12min41s| 1min5s                 | No        | No       |
| [``Snake-v1``](https://instadeepai.github.io/jumanji/environments/snake/)       | 80.000    | 12min9s | 15min                  | No        | Yes      |
| [``TSP-v1``](https://instadeepai.github.io/jumanji/environments/tsp/)           | 1.200.000 | 6min11s | 31s                    | Yes       | No       |

## Learning Curves

*Ces courbes sont fournies à titre indicatif pour des paramètres/architectures arbitraires qui nécessitent d'être optimisés.*
<div align="center">
  <div style="display: flex; justify-content: center;">
    <div style="margin-right: 20px;">
      <img src="https://raw.githubusercontent.com/RPegoud/jumanji_prioritised_dqn/main/images/maze.png" alt="Image 1"/>
      <p align="center"><em>Maze-v0</em></p>
    </div>
    <div style="margin-right: 20px;">
      <img src="https://raw.githubusercontent.com/RPegoud/jumanji_prioritised_dqn/main/images/knapsack.png" alt="Image 2"/>
      <p align="center"><em>Knapsack-v1</em></p>
    </div>
    <div>
      <img src="https://raw.githubusercontent.com/RPegoud/jumanji_prioritised_dqn/main/images/cvpr.png" alt="Image 3"/>
      <p align="center"><em>CVPR-v1<br>Capacitated Vehicle Routing Problem</em></p>
    </div>
  </div>
</div>
<div align="center">
  <div style="display: flex; justify-content: center;">
    <div style="margin-right: 20px;">
      <img src="https://raw.githubusercontent.com/RPegoud/jumanji_prioritised_dqn/main/images/tsp.png" alt="Image 1"/>
      <p align="center"><em>TSP-v1<br>Traveling Sales Person</em></p>
    </div>
    <div style="margin-right: 20px;">
      <img src="https://raw.githubusercontent.com/RPegoud/jumanji_prioritised_dqn/main/images/snake.png" alt="Image 2"/>
      <p align="center"><em>Snake-v1</em></p>
    </div>
  </div>
</div>

### ***Hyperparameters:***

```python
# Number of Training-Evaluation iterations
TRAINING_EVAL_ITERS = 120 # or 60 for environments requiring CNN

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
SEED = 42
NUM_ENVS = 8
BUFFER_SIZE = 10_000
ROLLOUT_LEN = 512
SGD_STEPS_PER_ROLLOUT = 64
TRAINING_ITERS = 20
TARGET_PERIOD = 10
AGENT_DISCOUNT = 0.99
EPSILON_INIT = 1.0
EPSILON_FINAL = 0.1
EPSILON_STEPS = 10_000
PRIORTIY_EXPONENT = 0.6
IMPORTANCE_SAMPLING_EXPONENT = 0.6

# Evaluation parameters
NUM_EVAL_EPISODES = 50
```

### ***DQN Architecture without CNN***

```python
def get_network_fn(num_outputs: int):
    def network_fn(obs: chex.Array) -> chex.Array:
        """Outputs action logits."""
        network = hk.Sequential(
            [
            hk.Linear(64),
            jax.nn.relu,
            hk.Linear(128),
            jax.nn.relu,
            hk.Linear(num_outputs),
            ]
        )
        return network(obs)

    return hk.without_apply_rng(hk.transform(network_fn))
```

### ***DQN Architecture with CNN***

```python
def get_network_fn(num_outputs: int):
    def network_fn(obs: chex.Array) -> chex.Array:
        """Outputs action logits."""
        network = hk.Sequential(
            [
            hk.Conv2D(32, kernel_shape=2, stride=1),
            jax.nn.relu,
            hk.Conv2D(32, kernel_shape=2, stride=1),
            jax.nn.relu,
            hk.Flatten(),
            hk.Linear(64),
            jax.nn.relu,
            hk.Linear(128),
            jax.nn.relu,
            hk.Linear(num_outputs),
            ]
        )
        return network(obs)

    return hk.without_apply_rng(hk.transform(network_fn))
```
