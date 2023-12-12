import numpy as np

"""
    Replay Buffer, used to store the most recent state details. This helps in avaiding memory leakage
"""
class Buffer:
    def __init__(self, buffer_size, state_size):

        self.current_index = 0
        self.buffer_size = buffer_size
        self.current_size = 0
        self.current_state = np.zeros([buffer_size,state_size])
        self.next_state = np.zeros([buffer_size,state_size])
        self.rewards = np.zeros(buffer_size)
        self.actions = np.zeros(buffer_size)
        self.done = np.zeros(buffer_size, dtype = bool)

    """
        Stores the state data in the replay buffer by handling the size requirements.
    """
    def store_data(self,step_data):
        current_state,next_state,action,reward,done = step_data
        self.current_state[self.current_index] = current_state
        self.next_state[self.current_index] = next_state
        self.rewards[self.current_index] = reward
        self.actions[self.current_index] = action
        self.done[self.current_index] = done
        if self.current_index + 1 > self.current_size:
            self.current_size = self.current_index + 1
        self.current_index = (self.current_index + 1) % self.buffer_size
       
    """
        for the given batch siize, randomly select the states in the replay buffer and retur it.
    """
    def get_batch_data(self,batch_size):
        buffer_indexes = np.random.randint(0,self.current_size,batch_size)
        batch_data = {}
        # print("buffer_indexes = ",buffer_indexes)

        batch_data["current_state"] = self.current_state[buffer_indexes]
        batch_data["next_state"] = self.next_state[buffer_indexes]
        batch_data["rewards"] = self.rewards[buffer_indexes]
        batch_data["actions"] = self.actions[buffer_indexes]
        batch_data["done"] = self.done[buffer_indexes]
        # print("batch_data[actions] = ",batch_data["actions"])
        return batch_data
