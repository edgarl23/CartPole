import math
import numpy as np
import random
import torch

class Acrobot:
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
        cos_theta1,sin_theta1,cos_theta2,sin_theta2,w_1,w_2=state
        tetha1=math.acos(cos_theta1)
        theta2=math.acos(cos_theta2)
        next_cos_theta1,next_sin_theta1,next_cos_theta2,next_sin_theta2,next_w_1,next_w_2=next_state
        next_tetha1=math.acos(next_cos_theta1)
        next_theta2=math.acos(next_cos_theta2)
        y=-cos_theta1-math.cos(tetha1+theta2)
        y_next=-next_cos_theta1-math.cos(next_tetha1+next_theta2)

        r=10*(y_next-y)
        if abs(w_2)>10:
            r-=0.3
        if y_next>1:
            r+=50
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