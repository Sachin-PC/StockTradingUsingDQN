import numpy as np
from collections import defaultdict
from typing import Callable, Tuple
import random
from feedForwardNetwork import *


def create_policy(epsilon: float, action_size, agent) -> Callable:
    """Creates a policy.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        epsilon: epsilon value
        action_size: number of actions
        agent: Neural Network agent
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """

    def get_action(state: Tuple) -> int:
        if np.random.rand() <= epsilon:
            return np.random.choice(action_size)
        act_values = predict(agent.nnModel, state)
        return np.argmax(act_values[0])

    return get_action

def create_epsilon_policy(Q: defaultdict, epsilon: float) -> Callable:
    """Creates an epsilon soft policy from Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon (float): softness parameter
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """
    num_actions = len(Q[0])


    def get_action(state: Tuple) -> int:
        
        state = tuple(state)
        if np.random.random() < epsilon:
            action = random.randint(0, num_actions-1)
        else:
            maxVal = -99999999
            # print("state = ",state)
            for vals in Q[state]:
                if vals > maxVal:
                    maxVal = vals
            t=0
            best_actions = []
            for vals in Q[state]:
                if vals == maxVal:
                    best_actions.append(t)
                t +=1
            num_best_actions = len(best_actions)
            action = best_actions[random.randint(0, num_best_actions-1)]
        return action

    return get_action