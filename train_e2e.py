import os
import glob
import time
from datetime import datetime

import torch
import math
import numpy as np

import gym
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

from PPO import PPO


def train():

    # hyperparameters 
    env_name = "finalEnvNew"
    
    has_continuous_action_space = False  # continuous action space; else discrete

    max_ep_len = 1500                   # max timesteps in one episode
    max_training_timesteps = int(6e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = 800                    # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.4                   # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)


    ################ PPO hyperparameters ################

    update_timestep = max_ep_len * 4     # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO, suggested in the paper
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)

    #####################################################



    print("training environment name : " + env_name)
    unity_env = UnityEnvironment( "./envs/finalEnvNew/kairos", worker_id=np.random.randint(0, 1000))
    env = UnityToGymWrapper( unity_env, flatten_branched=True )

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n



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
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)

    #####################################################


    ################### checkpointing ###################
    run_num_pretrained = run_num   

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    weight_to_load = 'env2'
    loading_path = directory + '/' + weight_to_load + '/' + "PPO_{}_{}_{}.pth".format(weight_to_load, random_seed, run_num_pretrained)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)







    ################################## 
    ########## TRAINING ##############
    ################################## 

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("=" * 60)


    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,success, mean_reward\n')
    time_step = 0
    i_episode = 0
    success_rate = []
    violation_rate = []
    mean_reward = []
    last_good = 0
    

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()

        for t in range(1, max_ep_len+1):
            first_dist = state[-1]
    
            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)

            new_distance = state[-1]

            #dense reward
            if (done and reward == 0.0):
                success_rate.append(reward)
                violation_rate.append(1)
                reward = -30.0
            elif (done and reward == 1.0):
                success_rate.append(reward)
                violation_rate.append(0)
                reward = 100.0
            else:
                reward = 10 *(first_dist - new_distance) #max(0,(first_dist - new_distance))
            
            mean_reward.append(reward)    
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
        
            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # break; if the episode is over
            if done:
                break
            
        # print average reward last 50 episodes and save weights if success is greater than 96%
        if i_episode % 100 == 0:
            print_avg_success = sum(success_rate[-100:])/100
            print_avg_success = round(print_avg_success, 2)

            print_avg_reward = sum(mean_reward[-100:])/100
            print_avg_reward = round(print_avg_reward, 2)


            print("Episode: {} \t Timestep: {} \t Success rate: {}%\t Mean reward: {} ".format(i_episode, time_step, print_avg_success*100, print_avg_reward))
            log_f.write('{},{},{},{}\n'.format(i_episode, time_step, print_avg_success,print_avg_reward))
            log_f.flush()

            
            if print_avg_success >= 0.7 and print_avg_success >= last_good:
                last_good = print_avg_success
                print("-"*40)
                print("Saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("Model saved!")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("-"*40)
        
            
        i_episode += 1


    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")




if __name__ == '__main__':
    train()
    
    
    
    
    
    
    
    
