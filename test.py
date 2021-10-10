import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper


# import pybullet_envs

from PPO import PPO



################################## 
########## Testing ###############
##################################


def test():
    device = torch.device('cpu')
    ################## hyperparameters #################
    env_name = "baseEnv"
    has_continuous_action_space = False
    max_ep_len = 1500           # max timesteps in one episode
    action_std = 0           # set same std for action distribution which was used while saving

    render = True              # render environment on screen
    total_test_episodes = 100    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic


    #####################################################



    print("testing environment name : " + env_name)

    unity_env = UnityEnvironment( f"./envs/{env_name}/kairos", worker_id=np.random.randint(0, 1000))
    env = UnityToGymWrapper( unity_env, flatten_branched=True)
 

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n


    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    
    

    # preTrained weights directory
    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    method = input('Select the training methodology to test:\n1 - E2E\n2 - TransferLearning \n3 - FineTuning\n\n')
    directory = "PPO_preTrained" + '/'
    if method == 'E2E':
        weights = directory + "PPO_E2E.pth"
    elif method == 'Transfer':
        weights = directory + "PPO_TransferLearning.pth"
    else:
        weights = directory + "PPO_FineTuning.pth"

    print("loading network from : " + weights)
    #weights = './PPO_preTrained/PPO_finalEnvNew_0_E2E.pth'
    ppo_agent.load(weights)
     
    for net in ppo_agent.policy.children():
        for layer in net.parameters():
            layer.requires_grad = False
        break
    print("-"*50)


    ###################### create log file ######################
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)


    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)


    #### create new log file for each run
    log_f_name = log_dir + '/PPO_Test_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,success,collision,avg_distances,method\n')


    for testEpisode in range(30):
        print(testEpisode)
        test_running_reward = 0
        test_running_collision_rate = 0
        distances = []
        episode_timesteps = []

        for ep in range(1, total_test_episodes+1):
            ep_reward = 0
            collision = 0
            state = env.reset()
            distances.append(state[-1])
            
            for t in range(1, max_ep_len+1):
               
                action = ppo_agent.select_action(state)
    
                state, reward, done, _ = env.step(action)
                ep_reward += reward
           
                if done:
                    if reward == 0:
                        collision = 1
                    episode_timesteps.append(t)
                    break

            # clear buffer
            ppo_agent.buffer.clear()
            test_running_collision_rate += collision
            test_running_reward +=  ep_reward
            ep_reward = 0

        
        #print("============================================================================================")
        avg_test_reward = test_running_reward / total_test_episodes
        avg_test_reward = round(avg_test_reward, 2)
        avg_test_collision = test_running_collision_rate/total_test_episodes
        avg_test_collision = round(avg_test_collision, 2)
        avg_distance = ((sum(episode_timesteps))/sum(distances))/total_test_episodes
        avg_distance = round(avg_distance, 2)
        log_f.write('{},{},{},{},{}\n'.format(testEpisode, avg_test_reward, avg_test_collision, avg_distance, method))
        log_f.flush()
        #print(f"average test reward : {avg_test_reward*100}%")
        #print(f"violation rate: {test_running_violation_rate}/{total_test_episodes}")
        #print("============================================================================================")
                
        avg_test_reward = 0
        avg_test_collision = 0
        avg_distance = 0

    env.close()


if __name__ == '__main__':
    test()
