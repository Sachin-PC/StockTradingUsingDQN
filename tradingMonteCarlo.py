import numpy as np
import os
from dQNAlgorithm import *
import pandas as pd
from sklearn.preprocessing import StandardScaler
from policy import *
from tqdm import trange
import argparse
from stockMarketEnvironment import *
import matplotlib.pyplot as plt
import pickle
from MCAlgorithms import *

  
  
def get_scaler(env):
  # used to scale the states
  states = []
  for _ in range(env.unique_stocks):
    action = np.random.choice(env.action_space)
    state, reward, done, info = env.step(action)
    states.append(state)
    if done:
      break

  scaler = StandardScaler()
  scaler.fit(states)
  return scaler


def plot_data(x_values,y_values,y_lable, title,labels):
    print("Inside")
    i=0
    for data in y_values:
        plt.plot(x_values,data, label = labels[i])
        i +=1
    plt.xlabel("Episodes")
    plt.ylabel(y_lable)
    plt.title(title)
    plt.legend()
    plt.show()

def rolling_average(data, *, window_size):
    """Smoothen the 1-d data array using a rollin average.

    Args:
        data: 1-d numpy.array
        window_size: size of the smoothing window

    Returns:
        smooth_data: a 1-d numpy.array with the same size as data
    """
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(
        np.ones_like(data), kernel
    )
    return smooth_data[: -window_size + 1]
  
if __name__ == '__main__':

    stockData   = pd.read_csv('stockData.csv').values
    number_of_records, unique_stocks = stockData.shape
    train_data_size = int(number_of_records/2)
    train_data = stockData[:train_data_size]
    test_data = stockData[train_data_size:]
    start_balance = 20000
    env = StockMarketEnv(train_data, start_balance)
    num_episodes = 5000
    gammas = [1, 0.9, 0.5, 0.2]
    # epsilons = [0.5,0.1,0.01]
    epsilon = 0.01
    # on_policy_mc_control_es(env,500000,1)
    final_training_rewards = []
    final_training_portfolio_values = []
    final_testing_rewards = []
    final_testing_portfolio_values = []
    for gamma in gammas:
        training_rewards, training_portfolio_values, Q = on_policy_mc_control_epsilon_soft(env, num_episodes, gamma, epsilon)
        env = StockMarketEnv(train_data, start_balance)
        testing_rewards, testing_portfolio_values= test_on_policy_mc_control_epsilon_soft(env, num_episodes, gamma, epsilon, Q)

        rolling_training_rewards = rolling_average(training_rewards,window_size = 100)
        rolling_training_portfolio_values = rolling_average(training_portfolio_values,window_size = 100)

        rolling_testing_rewards = rolling_average(testing_rewards,window_size = 100)
        rolling_testing_portfolio_values = rolling_average(testing_portfolio_values,window_size = 100)

        final_training_rewards.append(rolling_training_rewards)
        final_training_portfolio_values.append(rolling_training_portfolio_values)

        final_testing_rewards.append(rolling_testing_rewards)
        final_testing_portfolio_values.append(rolling_testing_portfolio_values)


    episodes = np.arange(0,num_episodes)
    for i in range(num_episodes):
        episodes[i] = i+1
    y_lable = "Returns"
    title = "Train Data Returns of each Episode"
    labels = ["gamma = 1", "gamma = 0.9", "gamma = 0.5", "gamma = 0.2"]
    plot_data(episodes,final_training_rewards, y_lable, title, labels)

    y_lable = "Portfolio Value"
    title = "Train Data Portfolio Value at end of each Episode"
    print("final_training_portfolio_values = ",final_training_portfolio_values)
    plot_data(episodes,final_training_portfolio_values, y_lable, title,labels)


    y_lable = "Returns"
    title = "Test Data Returns of each Episode"
    plot_data(episodes,final_testing_rewards, y_lable, title,labels)

    y_lable = "Portfolio Value"
    title = "Test Data Portfolio Value at end of each Episode"
    plot_data(episodes,final_testing_portfolio_values, y_lable, title,labels)
    print("444444")
