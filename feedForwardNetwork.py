import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

"""
    Neural Network model, used to train the action weights of the state
"""
class FeedNet(nn.Module):


    def __init__(self, input_size, hidden_layer_size, total_actions):
        super().__init__()
        self.flatten = nn.Flatten(start_dim = 1)
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_size, total_actions)

    def forward(self, x):

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        y_output = x
        return y_output
    
    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
    
def trainNNModel(nnModel, modelInputs, targets, criterion, optimizer):
    modelInputs = torch.from_numpy(modelInputs.astype(np.float32))
    targets = torch.from_numpy(targets.astype(np.float32))

    output = nnModel(modelInputs)
    loss = criterion(output, targets)
            
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def predict(nnModel, modelInputs):
    with torch.no_grad():
        modelInputs = torch.from_numpy(modelInputs.astype(np.float32))
        output = nnModel(modelInputs)
        return output.numpy()