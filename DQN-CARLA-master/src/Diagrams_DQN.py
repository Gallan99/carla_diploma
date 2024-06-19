import matplotlib.pyplot as plt
import numpy as np



N=10
average_reward_list=[]
max_reward_list=[]
min_reward_list=[]
average_dist_list=[]
average_episodes_list=[]

with open("C:/Users\galag\OneDrive\Diploma\DQN-CARLA-master\src\episodes_info\episodes.txt") as f:
    episodes_list = [int(x) for x in f.read().split()]
with open("C:/Users\galag\OneDrive\Diploma\DQN-CARLA-master\src\episodes_info\episodes_dist.txt") as f:
    episodes_dist_list = [float(x) for x in f.read().split()]
with open("C:/Users\galag\OneDrive\Diploma\DQN-CARLA-master\src\episodes_info\episodes_reward.txt") as f:
    episodes_reward_list = [float(x) for x in f.read().split()]




for i in range(0,len(episodes_list)):
    if (i>1) and (i % N == 0):
        average_episodes_list.append(i)
        max_reward_list.append(max(episodes_reward_list[i-10:i]))
        min_reward_list.append(min(episodes_reward_list[i-10:i]))
        average_reward_list.append(sum(episodes_reward_list[i-10:i])/len(episodes_reward_list[i-10:i]))
        average_dist_list.append(sum(episodes_dist_list[i-10:i])/len(episodes_reward_list[i-10:i]))


x1 = np.array(episodes_list)
x2 = np.array(average_episodes_list)
y1 = np.array(episodes_reward_list)
y2 = np.array(min_reward_list)
y3 = np.array(max_reward_list)
y4 = np.array(average_dist_list)
y5 = np.array(average_reward_list)
y6 = np.array(episodes_dist_list)


plt.plot(x1, y1, color="green")
plt.title("Reward-Episodes_DQN")
plt.savefig('Reward-Episodes_8000_DQN.png')
plt.close()
plt.plot(x2, y2, color="green")
plt.title("Minimum Reward-Episodes_DQN")
plt.savefig('Minimum Reward-Episodes_8000_DQN.png')
plt.close()
plt.plot(x2, y3, color="green")
plt.title("Maximum Reward-Episodes_DQN")
plt.savefig('Maximum Reward-Episodes_8000_DQN.png')
plt.close()
plt.plot(x2, y4, color="green")
plt.title("Average Distance-Episodes_DQN")
plt.savefig('Average Distance-Episodes_8000_DQN.png')
plt.close()
plt.plot(x2, y5, color="green")
plt.title("Average Reward-Episodes_DQN")
plt.savefig('Average Reward-Episodes_8000_DQN.png')
plt.close()
plt.plot(x1, y6, color="red")
plt.title("Distance-Episodes_DDPG")
plt.savefig('Distance-Episodes_8000_DDPG.png')
plt.close()