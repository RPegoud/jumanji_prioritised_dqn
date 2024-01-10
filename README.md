# Solving Jumanji Environments with Prioritised DQN

| Environment                                                                     | Timesteps | Time    | Time per 100,000 steps | Converged | Uses CNN |
|:------------------------------------------------------------------------------- |:--------- | ------- | ---------------------- | --------- |:-------- |
| [``CVRP-v1``](https://instadeepai.github.io/jumanji/environments/cvrp/)         | 1.200.000 | 7min12s | 36s                    | No        | No       |
| [``Knapsack-v1``](https://instadeepai.github.io/jumanji/environments/knapsack/) | 1.200.000 | 8min20s | 43s                    | Yes       | No       |
| [``Maze-v0``](https://instadeepai.github.io/jumanji/environments/maze/)         | 1.200.000 | 12min41s| 1min5s                 | No        | No       |
| [``Snake-v1``](https://instadeepai.github.io/jumanji/environments/snake/)       | 80.000    | 12min9s | 15min                  | No        | Yes      |
| [``TSP-v1``](https://instadeepai.github.io/jumanji/environments/tsp/)           | 1.200.000 | 6min11s | 31s                    | Yes       | No       |

## Learning Curves

<div align="center">
  <div style="display: flex; justify-content: center;">
    <div style="margin-right: 20px;">
      <img src="https://raw.githubusercontent.com/RPegoud/jumanji_prioritised_dqn/main/images/maze.png" alt="Image 1"/>
      <p align="center"><em>Maze-v0</em></p>
    </div>
    <div style="margin-right: 20px;">
      <img src="https://raw.githubusercontent.com/RPegoud/jumanji_prioritised_dqn/main/images/knapsack.png" alt="Image 2"/>
      <p align="center"><em>``Knapsack``</em></p>
    </div>
    <div>
      <img src="https://raw.githubusercontent.com/RPegoud/jumanji_prioritised_dqn/main/images/cvpr.png" alt="Image 3"/>
      <p align="center"><em>Dyna</em></p>
    </div>
  </div>
</div>
<div align="center">
  <div style="display: flex; justify-content: center;">
    <div style="margin-right: 20px;">
      <img src="https://raw.githubusercontent.com/RPegoud/jumanji_prioritised_dqn/main/images/tsp.png" alt="Image 1"/>
      <p align="center"><em></em></p>
    </div>
    <div style="margin-right: 20px;">
      <img src="https://raw.githubusercontent.com/RPegoud/jumanji_prioritised_dqn/main/images/snake.png" alt="Image 2"/>
      <p align="center"><em></em></p>
    </div>
  </div>
</div>
