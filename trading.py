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


def generate_episode(env, policy: Callable, agent, evaluationType):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples


    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        agent:
        mode: 
    """
    average_reward = 0
    state = env.reset()
    state = scaler.transform([state])
    count = 0
    batch_size = 10
    while True:
        action = policy(state)
        next_state, reward, done, p_val = env.step(action)
        next_state = scaler.transform([next_state])
        average_reward += reward
        count += 1
        if evaluationType == 'train':
            step_data = [state,next_state, action, reward, done]
            agent.buffer_update(step_data)
            agent.q_learn(batch_size)
        if done:
           break
        state = next_state

    average_reward = average_reward/count

    return p_val, average_reward

def plot_data(x_values,y_values,y_lable, title):
    plt.plot(x_values,y_values)
    plt.xlabel("Episodes")
    plt.ylabel(y_lable)
    plt.title(title)
    plt.legend()
    plt.show()
  
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--evaluation')
    args = parser.parse_args()
    num_episodes = 1000
    batch_size = 32
    gamma = 0.9
    epsilon = 0.1
    bufferSize = 1000

    folder = "nnWeightsData"
    if not os.path.exists("nnWeightsData"):
        os.makedirs(folder)

    stockData   = pd.read_csv('stockData.csv').values
    number_of_records, unique_stocks = stockData.shape
    train_data_size = int(number_of_records/2)
    train_data = stockData[:train_data_size]
    test_data = stockData[train_data_size:]

    start_balance = 20000
    env = StockMarketEnv(train_data, start_balance)
    
    hidden_layer_size = 32
    nnModel = FeedNet(env.state_dim, hidden_layer_size, len(env.action_space))
    agent = DQNAgent(env,nnModel,gamma,epsilon,env.state_dim,bufferSize)
    policy = create_policy(epsilon,len(env.action_space),agent)
    end_balance = []
    average_rewards = []
    scaler = get_scaler(env)
    if args.evaluation == 'test':
        nnModel.load_weights(f'{folder}/agent.ckpt')
        with open(f'{folder}/scalerData.pkl', 'rb') as f:
            scaler = pickle.load(f)
        env = StockMarketEnv(test_data, start_balance)
        agent = DQNAgent(env,nnModel,gamma,epsilon,env.state_dim,bufferSize)

    for _ in trange(num_episodes, desc="Episode"):
        p_val, average_reward = generate_episode(env,policy,agent,args.evaluation)
        end_balance.append(p_val)
        average_rewards.append(average_reward)
    episodes = np.arange(0,num_episodes)
    for i in range(len(episodes)):
        episodes[i] = i+1
    y_lable = "Portfolio Value"
    title = "Portfolio value at end of each Episode"
    plot_data(episodes,end_balance, y_lable, title)
    y_lable = "Average Rewards"
    title = "Average Rewards/Episode"
    plot_data(episodes,average_rewards, y_lable, title)

    if args.evaluation == 'train':
        nnModel.save_weights(f'{folder}/agent.ckpt')
        with open(f'{folder}/scalerData.pkl', 'wb') as f:
            pickle.dump(scaler, f)