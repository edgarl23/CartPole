import torch
import random
import numpy as np
import math

class MountainCar:
    def __init__(self, env, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999):
        self.env = env
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
    
    def get_action(self,q_values):

        sample=np.random.rand()
        if sample<self.epsilon:
            return random.choice(range(len(q_values)))

        self.epsilon=max(self.epsilon*self.epsilon_decay,self.epsilon_min)
        return torch.argmax(q_values).item()

    def reward_new(self,state,next_state,done):
        x,v=state
        next_x,next_v=next_state
        h_next=math.sin(3*next_x)
        h=math.sin(3*x)
        #r=(next_x-x)*2
        r=80*(h_next-h)+140*(next_v-v)
        if done and next_state[0]>=0.5:
            r+=100
        if (h_next-h)==0:
            r-=0.1
        return r
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
    
    
    def reset(self):
        self.env.reset()



