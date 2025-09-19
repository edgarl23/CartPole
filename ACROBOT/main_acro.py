import gym
from Qnetworks import Qnetwork,update_target_network
import torch.optim as optim
from agent import Acrobot
from Replay_buffer import ReplayBuffer
import torch
import torch.functional as F




env_acrobot = gym.make("Acrobot-v1", render_mode=None)  
replay_acrobot_buffer = ReplayBuffer(50000)
tau_acrobot=0.001
gamma_acrobot=0.99
batch_size_acrobot=128
target_network_acrobot=Qnetwork(6,3)
online_network_acrobot=Qnetwork(6,3)
optimizer_acrobot=optim.Adam(online_network_acrobot.parameters(),lr=5e-4)
acrobot=Acrobot(env_acrobot)
acrobot.set_seed(seed=20)
reward_list=[]
epsilon=1.0
end=0.01
decay=0.995


for episode in range(2000):

    state,info=env_acrobot.reset()
    done=False
    step=0
    episode_reward=0
    episode_plus_reward=0

    while not done:
        step+=1
        q_values=online_network_acrobot(state=torch.tensor(state))
        action=acrobot.get_action(q_values)
        next_state,reward,terminate,truncated,_=env_acrobot.step(action)
        r=acrobot.reward_new(state,next_state,done)
        reward=reward+r
        done=terminate or truncated
        replay_acrobot_buffer.push(state,action,reward,next_state,done)

        if len(replay_acrobot_buffer)>=batch_size_acrobot:

            states,actions,rewards,nex_states,dones=replay_acrobot_buffer.sample(batch_size_acrobot)
            q_values=online_network_acrobot(states).gather(1,actions).squeeze(1)
            with torch.no_grad():
                next_actions=online_network_acrobot(nex_states).argmax(1).unsqueeze(1)
                next_q_values=target_network_acrobot(nex_states).gather(1,next_actions).squeeze(1)
                target_q_values=rewards+gamma_acrobot*next_q_values*(1-dones)

            loss=F.mse_loss(q_values, target_q_values)
            optimizer_acrobot.zero_grad()
            loss.backward()
            optimizer_acrobot.step()
            update_target_network(target_network_acrobot,online_network_acrobot,tau_acrobot)
        
        state=next_state
        episode_reward+=reward
        episode_plus_reward+=r
    reward_list.append(episode_reward)
    
    print(f"Episode: {episode}, Step: {step},reward_new:{round(r,3)}, Total Reward: {round(episode_reward,2)}")