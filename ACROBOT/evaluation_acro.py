import gym
import torch
from main import online_network_acrobot
import math 


def evaluate(state):
        cos_theta1,sin_theta1,cos_theta2,sin_theta2,w_1,w_2=state
        tetha1=math.acos(cos_theta1)
        theta2=math.acos(cos_theta2)
        y=-cos_theta1-math.cos(tetha1+theta2)

        return y

completados=0
step_list=[]
env_acrobot = gym.make("Acrobot-v1", render_mode=None) 
for episode in range(1000):

    state,info=env_acrobot.reset()
    done=False
    step=0


    while not done:

        with torch.no_grad():
            step+=1
            q_values=online_network_acrobot(torch.tensor(state))
            action = torch.argmax(q_values).item()
            next_state,reward,terminate,truncated,_=env_acrobot.step(action)
            done=terminate or truncated

        state=next_state

        y=evaluate(next_state)
        if done and y>1:
            completados+=1
            print(f"Episodio {episode+1},step:{step} completado--- completados: {completados}")
    
    step_list.append(step)

env_acrobot.close()