from buffer import *
from feedForwardNetwork import *
import gym
import torch.optim as optim

"""
  This class is defined as  Deep Q Network agent which uses a neural netowkr model
  to train the weights of the states actions and hence, helps in choosing the best action.
"""
class DQNAgent(object):
  def __init__(self,env,nnModel,gamma,epsilon,state_dim,bufferSize = 1000):
    self.nnModel = nnModel
    self.gamma = gamma
    # self.epsilon = epsilon
    self.buffer = Buffer(bufferSize,state_dim)
    self.criterion = torch.nn.CrossEntropyLoss()
    self.optimizer = optim.SGD(self.nnModel.parameters(), lr=0.01)
    self.epsilon = 1.0  # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995


  def buffer_update(self, step_data):
    """
    updates the replay buffer with the given single step data
    """
    self.buffer.store_data(step_data)


  def q_learn(self, batch_size=32):
    """
        This method is used to do the q_learing by first calculating the target 
        by q update and the using it to train the neural network model.
    """
    # first check if replay buffer contains enough data
    if self.buffer.current_size < batch_size:
      return

    # sample a batch of data from the replay memory
    minibatch = self.buffer.get_batch_data(batch_size)
    states = minibatch['current_state']
    actions = minibatch['actions']
    rewards = minibatch['rewards']
    next_states = minibatch['next_state']
    done = minibatch['done']

    # with torch.no_grad():
    # #     # Select the actions that maximize the Q-values for the next state
    #     # max_next_actions = self.nnModel(next_states).argmax(dim=1, keepdim=True)
    # #     # Compute target Q-values using the target DQN model and Bellman equation
    # #     target_values = rewards + self.gamma * self.nnModel(next_states).gather(1, max_next_actions) * (1 - done)

    # Calculate Q(s',a)
    predictedValue = predict(self.nnModel, next_states)
    target = rewards + (1 - done) * self.gamma * np.amax(predictedValue, axis=1)

    target_full = predict(self.nnModel, states)
    target_full[np.arange(batch_size), actions.astype(int)] = target

    trainNNModel(self.nnModel, states, target_full, self.criterion, self.optimizer)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay