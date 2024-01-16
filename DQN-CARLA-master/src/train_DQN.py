import os
import sys
import random
import time
import numpy as np
import cv2
import math
from datetime import date
import matplotlib.pyplot as plt

import tensorflow as tf
#from keras.backend import set_session as backend
import keras.backend.tensorflow_backend as backend
from threading import Thread

from tqdm import tqdm


import carla_config as settings
from agent_model import DQNAgent
from carla_env import CarEnv


# Own Tensorboard class

if __name__ == '__main__':
    distance_acum = []
    epsilon = settings.epsilon
    FPS = 20
    # For stats
    ep_rewards = [-200]
    # tf.config.optimizer.set_jit(True)
    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.compat.v1.set_random_seed(1)
    # Memory fraction, used mostly when training multiple agents
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
    backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))
    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')
    # print("Antes de create agent")
    # Create agent and environment
    agent = DQNAgent()
    env = CarEnv()
    # print("Despues de agente y environment")
    date_title = date.today()
    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()

    while not agent.training_initialized:
        # print("Esperando inicializacion de agente")
        # Waiting for agent initialization
        time.sleep(0.01)
    # Before get_qs"
    # print("Antes de get_qs")

    # Initialize predictions - first prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    # agent.get_qs(np.ones((env.im_height, env.im_width, IM_LAYERS)))
    # making first Q-values prediction
    if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or \
            settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] or \
            settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7] or \
            settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
        agent.get_qs(np.ones(settings.state_dim, ))
    else:
        agent.get_qs(np.ones((settings.IM_HEIGHT_CNN, settings.IM_WIDTH_CNN, settings.IM_LAYERS)))
    max_rewrd_list=[]
    min_reward_list=[]
    avg_reward_list=[]
    avg_dist_list=[]
    episode_list=[]
    # Iterate over episodes
    for episode in tqdm(range(1, settings.EPISODES + 1), ascii=True, unit='episodes'):
        # try:
        env.collision_hist = []
        env.crossline_hist = []

        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or\
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] or\
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
            # we take state train from transform2local
            _, state_train = env.reset
        else:
            state_train, _ = env.reset
            if settings.IM_LAYERS == 1:
                state_train = np.expand_dims(state_train, -1)
            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7]:
                state_train = state_train.flatten()


        done = False
        episode_start = time.time()
        # Play for given number of seconds only

        while True:

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                # perform exploitation
                # Get action from Q table
                action_vector = agent.get_qs(state_train)
                # returns the highest Q-value
                action = np.argmax(action_vector)
                #print(settings.ACTIONS_NAMES[action])
                # print(settings.ACTIONS_NAMES[action])

            else:
                # perform exploration
                # Get random action
                # print("Accion random")
                action = np.random.randint(0, settings.N_actions)
                # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                time.sleep(1 / FPS)
            # if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0]:
            #     [_, new_state_train], reward, done, _ = env.step(action)
            # elif settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1]:
            #     new_image, reward, done, info = env.step(action)
            #     new_state_train = env.Calcular_estado(new_image)
            # we take new_state_train from transform2local
            if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[0] or \
                    settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[1] or\
                    settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[8] or \
                settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[9]:
                [_, new_state_train], reward, done, _ = env.step(action)
            else:
                [new_state_train, _], reward, done, _ = env.step(action)
                if settings.IM_LAYERS == 1:
                    new_state_train = np.expand_dims(new_state_train, -1)
                if settings.WORKING_MODE == settings.WORKING_MODE_OPTIONS[7]:
                    new_state_train = new_state_train.flatten()

            #print('Action: ', ACTIONS_NAMES[action], ' Reward: ', reward)

            # Transform new continous state to new discrete state and count reward
            # in every step we sum the rewards
            episode_reward += reward

            # Every step we update replay memory
            agent.update_replay_memory((state_train, action, reward, new_state_train, done))


            state_train = new_state_train
            step += 1
            if done:
                break

        print('episode reward:',episode_reward)
        #print(agent.model.get_weights())
        #json_wei = agent.model.to_json()
        #print(json_wei)


        # End of episode - destroy agents
        for actor in env.actor_list:
            actor.destroy()


        # Decay epsilon
        if epsilon > settings.MIN_EPSILON:
            epsilon *= settings.EPSILON_DECAY
            epsilon = max(settings.MIN_EPSILON, epsilon)

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if (episode > 1) and ((episode % settings.AGGREGATE_STATS_EVERY) == 0) or (episode == 2):
            # we sum the rewards from the last 10 episode
            average_reward = sum(ep_rewards[-settings.AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-settings.AGGREGATE_STATS_EVERY:]) # avg reward from the last 10 episodes
            min_reward = min(ep_rewards[-settings.AGGREGATE_STATS_EVERY:]) # min reward from the last 10 episodes
            max_reward = max(ep_rewards[-settings.AGGREGATE_STATS_EVERY:]) # max reward from the last 10 episodes
            # we sum the distances from the last 10 episode
            average_dist = sum(env.distance_acum[-settings.AGGREGATE_STATS_EVERY:]) / len(env.distance_acum[-settings.AGGREGATE_STATS_EVERY:]) # avg distance from the last 10 episodes
            avg_reward_list.append(average_reward)
            min_reward_list.append(min_reward)
            max_rewrd_list.append(max_reward)
            avg_dist_list.append(average_dist)
            episode_list.append(episode)
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           efshowpsilon=epsilon, avegare_dist=average_dist)
            print('avg reward:',average_reward)
        # Guardar datos del entrenamiento en ficheros
        # Save training data to files
        if episode % 3 == 0:
            agent.model.save(settings.AGENT_PATH + str(settings.TRAIN_MODE)+"_model.model")
        if episode % settings.N_save_stats == 0:
            agent.model.save(settings.AGENT_PATH + str(settings.TRAIN_MODE)+"_" + str(episode) + "_model.model")
        if (episode > 10) and (episode_reward > np.max(ep_rewards[:-1])):
            agent.model.save(settings.AGENT_PATH + str(settings.TRAIN_MODE)+"_best_reward_model.model")


        acum = 0

    file1 = open('200ep/avg_reward_list_200', 'w')
    file2 = open('200ep/min_reward_list_200', 'w')
    file3 = open('200ep/max_reward_list_200', 'w')
    file4 = open('200ep/average_dist_list_200', 'w')
    file5 = open('200ep/episode_200', 'w')
    for i in range(0, len(avg_dist_list)):
        file1.write("%s\n" % avg_reward_list[i])
        file2.write("%s\n" % min_reward_list[i])
        file3.write("%s\n" % max_rewrd_list[i])
        file4.write("%s\n" % avg_dist_list[i])
        file5.write("%s\n" % episode_list[i])
    file1.close()
    file2.close()
    file3.close()
    file4.close()
    file5.close()


    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(settings.AGENT_PATH + str(settings.TRAIN_MODE) + "_last_model.model")