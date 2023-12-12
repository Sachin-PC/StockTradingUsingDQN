import gym
from typing import Callable, Tuple
from collections import defaultdict
from tqdm import trange
import numpy as np
from policy import create_epsilon_policy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


def generate_episode(env: gym.Env, policy: Callable, es: bool = False):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """
    episode = []
    state = env.reset()
    count =0
    while True:
        if es and len(episode) == 0:
            # action = env.action_space.sample()
            action = policy(state)
        else:
            action = policy(state)
        next_state, reward, done, p_val = env.step(action)
        episode.append((tuple(state), action, reward))
        if done:
            # print("done")
            break
        if count == 500:
            break
        state = next_state
        count +=1

    return episode, p_val



def on_policy_mc_control_epsilon_soft(
    env: gym.Env, num_episodes: int, gamma: float, epsilon: float
):
    """On-policy Monte Carlo policy control for epsilon soft policies.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
        epsilon (float): Parameter for epsilon soft policy (0 <= epsilon <= 1)
    Returns:

    """
    Q = defaultdict(lambda: np.zeros(27))
    N = defaultdict(lambda: np.zeros(27))

    policy = create_epsilon_policy(Q, epsilon)

    returns = np.zeros(num_episodes)
    portfolio_values = np.zeros(num_episodes)
    for i in trange(num_episodes, desc="Episode", leave=False):
        episode, p_val = generate_episode(env, policy, es=True)
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma*G + reward
            if (state, action) not in [(e[0], e[1]) for e in episode[:t]]:
                prev_num_state_samples = N[state][action]
                if prev_num_state_samples == 0: 
                    Q[state][action] = G
                else:
                    state_prev_average_reward = Q[state][action]
                    state_new_reward = (G + (prev_num_state_samples*state_prev_average_reward))/(prev_num_state_samples + 1)
                    Q[state][action] = state_new_reward
                N[state][action] = prev_num_state_samples + 1
        returns[i] = G
        portfolio_values[i] = p_val
    return returns, portfolio_values, Q