# Code for "Principled Exploration via Optimistic Bootstrapping and Backward Induction"

## Prerequisites
- Tensorflow-gpu > 1.13 with eager execution, or tensorflow 2.x
- Tensorflow-probability 0.6.0
- OpenAI [baselines](https://github.com/openai/baselines)
- OpenAI [Gym](http://gym.openai.com/)

## Implemented Algorithms

### Basic Algorithm

- Bootstrapped EBU (The basic algorithm of BEBU-UCB, BEBU-IDS and OB2I)
- [Bootstrapped DQN](https://arxiv.org/abs/1602.04621)
- [EBU](https://arxiv.org/abs/1805.12375) 
(we provide a clean implementation at https://github.com/review-anon/EBU)

### Action Selection

- [Thomson sampling](https://arxiv.org/abs/1602.04621)
- [Ensemble Vote](https://arxiv.org/abs/1706.01502)
- [UCB Exploration](https://arxiv.org/abs/1706.01502)
- [Information-Directed Sampling (IDS)](https://arxiv.org/abs/1812.07544)

### Bonus

- UCB-Bonus

### Others

- [Randomized Prior Function](https://arxiv.org/abs/1806.03335) 

## Usage

### Run OB2I

The following command should train an agent on "Breakout".

`python run_atari.py --env BreakoutNoFrameskip-v4 --reward-type ucb --ebu`

### Run Other Baselines

The following commands should train an agent on "Breakout" with other baselines.

#### Bootstrapped EBU (BEBU)

`python run_atari.py --env BreakoutNoFrameskip-v4 --ebu`

#### Bootstrapped EBU + UCB action-selection (BEBU-UCB)

`python run_atari.py --env BreakoutNoFrameskip-v4 --action-selection ucb --ebu`

#### Bootstrapped EBU + IDS action-selection (BEBU-IDS)

`python run_atari.py --env BreakoutNoFrameskip-v4 --action-selection ids --ebu`

#### Bootstrapped EBU + Ensemble Vote 

(vote is used for evaluation)

`python run_atari.py --env BreakoutNoFrameskip-v4 --action-selection vote --ebu`

#### Bootstrapped DQN

`python run_atari.py --env BreakoutNoFrameskip-v4`

#### Bootstrapped DQN + Ensemble Vote

`python run_atari.py --env BreakoutNoFrameskip-v4 --action-selection vote`

#### Bootstrapped DQN + UCB action-selection

`python run_atari.py --env BreakoutNoFrameskip-v4 --action-selection ucb`

#### Bootstrapped DQN + IDS action-selection

`python run_atari.py --env BreakoutNoFrameskip-v4 --action-selection ids`

#### Randomized Prior Function

Any method can combine with the Randomized Prior Function by using `--prior` flag.

For example, run Bootstrapped DQN + Randomized Prior Function as

`python run_atari.py --env BreakoutNoFrameskip-v4 --prior`


## Structure Overview

- `deepq.py` contains stepping the environment, storing experience and saving models.
- `deepq_learner.py` contains action-selection methods, bonus, bootstrapped DQN/EBU training.
- `replay_buffer.py` contains two class of replay buffer for BDQN and BEBU, respectively. The memory consumption has been highly optimized.
- `models.py` contains Q-network, Bootstrapped Q-network with multiple heads, Bootstrapped Q-network with Randomized Prior Function.
- `run_atari.py` contains hyper-parameters setting. Run this file will start training.


## Execution

The data for separate runs is stored on disk under the `result` directory with filename 
`<env-id>-<algorithm>-<date>-<time>.` Each run directory contains
- `log.txt` Record the episode, exploration rate, episodic rewards in training 
(after normalization and used for training), episodic scores (raw scores), current timesteps, percentage completed.
- `monitor.csv` Env monitor file by using `logger` from `Openai Baselines`.
- `parameters.txt` All hyper-parameters used in training.
- `progress.csv` Same data as `log.txt` but with `csv` format.
- `evaluate scores.txt` Evaluation of policy for 108000 frames every 1e5 training steps with 30 no-op evaluation. 
- `model_10M.h5`, `model_20M.h5`, `model_best_10M.h5`, `model_best_20M.h5` are the policy files saved.
