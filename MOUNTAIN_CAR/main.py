from Replay_buffer import ReplayBuffer,update_target_network
import gym
from Qnetworks import Qnetwork
import torch.optim as optim
from Agent import MountainCar
import torch
import torch.functional as F





env_car = gym.make("MountainCar-v0", render_mode=None)  
replay_car_buffer = ReplayBuffer(50000)
tau_car=0.001
gamma_car=0.99
batch_size_car=128
target_network_car=Qnetwork(2,3)
online_network_car=Qnetwork(2,3)
optimizer_car=optim.Adam(online_network_car.parameters(),lr=5e-4)
mountaincar=MountainCar(env_car)
mountaincar.set_seed(seed=15)
reward_list=[]
epsilon=1.0
end=0.01
decay=0.995


for episode in range(1000):

    state,info=env_car.reset()
    done=False
    step=0
    episode_reward=0
    episode_plus_reward=0

    while not done:
        step+=1
        q_values=online_network_car(state=torch.tensor(state))
        action=mountaincar.get_action(q_values)
        next_state,reward,terminate,truncated,_=env_car.step(action)
        r=mountaincar.reward_new(state,next_state,done)
        reward=reward+r
    
        done=terminate or truncated
        replay_car_buffer.push(state,action,reward,next_state,done)

        if len(replay_car_buffer)>=batch_size_car:

            states,actions,rewards,next_states,dones=replay_car_buffer.sample(batch_size_car)
    
            q_values=online_network_car(states).gather(1,actions).squeeze(1)
            with torch.no_grad():
                next_actions=online_network_car(next_states).argmax(1).unsqueeze(1)
                next_q_values=target_network_car(next_states).gather(1,next_actions).squeeze(1)
                target_q_values=rewards+gamma_car*next_q_values*(1-dones)

            loss=F.mse_loss(q_values, target_q_values)
            optimizer_car.zero_grad()
            loss.backward()
            optimizer_car.step()
            update_target_network(target_network_car,online_network_car,tau_car)
        
        state=next_state
        episode_reward+=reward
        episode_plus_reward+=r
    reward_list.append(episode_reward)
    
    print(f"Episode: {episode}, Step: {step},reward_new:{round(episode_plus_reward,3)}, Total Reward: {round(episode_reward,2)}")