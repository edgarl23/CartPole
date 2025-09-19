import torch.nn as nn
import torch

class Qnetwork(nn.Module):
    def __init__(self,state_size,action_size):
        super(Qnetwork,self).__init__()

        self.fc1=nn.Linear(state_size,128)
        self.fc2=nn.Linear(128,128)
        self.fc3=nn.Linear(128,action_size)

    def forward(self,state):

        x=torch.relu(self.fc1(state))
        x=torch.relu(self.fc2(x))
        return self.fc3(x)

def update_target_network(target_network,online_network,tau):
    target_net_state_dict=target_network.state_dict()
    online_net_state_dict=online_network.state_dict()

    for key in online_net_state_dict:
        target_net_state_dict[key]=(online_net_state_dict[key]*tau+target_net_state_dict[key]*(1-tau))

    target_network.load_state_dict(target_net_state_dict)
        
    return done