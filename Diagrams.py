import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



with open("episodes_info/episodes") as f:
    episode_list = [int(x) for x in f.read().split()]
with open("episodes_info/max_reward_list") as f:
    max_reward_list = [float(x) for x in f.read().split()]
with open("episodes_info/average_dist_list") as f:
    average_dist_list = [float(x) for x in f.read().split()]
with open("episodes_info/avg_reward_list") as f:
    avg_reward_list = [float(x) for x in f.read().split()]
with open("episodes_info/min_reward_list") as f:
    min_reward_list = [float(x) for x in f.read().split()]


    x = np.array(episode_list)
    y1 = np.array(avg_reward_list)
    y2 = np.array(min_reward_list)
    y3 = np.array(max_reward_list)
    y4 = np.array(average_dist_list)

    plt.subplot(2, 2, 1)
    plt.plot(x, y1, color="red")
    plt.title("Average Reward-Episodes_8000")
    plt.subplot(2, 2, 2)
    plt.plot(x, y2, color="blue")
    plt.title("Minimum Reward-Episodes_8000")
    plt.subplot(2, 2, 3)
    plt.plot(x, y3, color="green")
    plt.title("Maximum Reward-Episodes_8000")
    plt.subplot(2, 2, 4)
    plt.plot(x, y4, color="yellow")
    plt.title("Average Distance-Episodes_8000")
    plt.subplots_adjust(left=0.12,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.5,
                            hspace=0.5)
    plt.savefig('episodes_stats_8000.png')
