from collections import deque
import math
import random
import torch

class ReplayBuffer:

    def __init__(self,capacity):

        self.memory=deque(maxlen=capacity)
    
    def push(self,state,action,reward,next_state,done):

        experience_tuple=(state,action,reward,next_state,done)
        self.memory.append(experience_tuple)

    def __len__(self):
        return  len(self.memory)
    
    def sample(self,batch_size):

        batch=random.sample(self.memory,batch_size)

        states,actions,rewards,next_states,dones=zip(*batch)
        
        states=torch.tensor(states,dtype=torch.float32)
        actions=torch.tensor(actions,dtype=torch.long).unsqueeze(1)
        rewards=torch.tensor(rewards,dtype=torch.float32)
        next_states=torch.tensor(next_states,dtype=torch.float32)
        dones=torch.tensor(dones,dtype=torch.float32)

        return(states,actions,rewards,next_states,dones)