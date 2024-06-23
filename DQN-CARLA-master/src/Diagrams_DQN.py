import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics
import scipy.stats as st



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

# rolling mean for rewards and mean
pd_reward = pd.DataFrame(episodes_reward_list)
rolling_mean_reward = pd_reward.rolling(500).mean()
pd_dist = pd.DataFrame(episodes_dist_list)
rolling_mean_dist = pd_dist.rolling(500).mean()

# Calculate standard deviation and print for rewards and dist
std_reward_DQN = statistics.stdev(episodes_reward_list)
std_dist_DQN = statistics.stdev(episodes_dist_list)
print("Standard deviation of DQN reward: " + str(std_reward_DQN))
print("Standard deviation of DQN dist: " + str(std_dist_DQN))

# Calculate 25% and 50% percentiles for rewards and dist
per50_rewards_DQN = np.percentile(episodes_reward_list,50)
per25_rewards_DQN = np.percentile(episodes_reward_list,25)
per50_dist_DQN = np.percentile(episodes_dist_list,50)
per25_dist_DQN = np.percentile(episodes_dist_list,25)
print("50th percentile of DQN rewards: ", per50_rewards_DQN)
print("25th percentile of DQN rewards: ", per25_rewards_DQN)
print("50th percentile of DQN dist: ", per50_dist_DQN)
print("25th percentile of DQN dist: ", per25_dist_DQN)

# Calculate the 95% confidence Intervals of rewards and dist
sample_mean_r = np.mean(episodes_reward_list)
sample_std_r = np.std(episodes_reward_list, ddof=1)
sample_mean_d = np.mean(episodes_dist_list)
sample_std_d = np.std(episodes_dist_list, ddof=1)
n=len(episodes_reward_list)
cl=0.95
alpha = 1- cl
z= st.norm.ppf(1-alpha/2)
error = z*(sample_std_r/np.sqrt(n))
ci_r = (sample_mean_r-error,sample_mean_r+error)
error = z*(sample_std_d/np.sqrt(n))
ci_d = (sample_mean_d-error,sample_mean_d+error)
print("95% confidence intervals of rewards are between: ", ci_r)
print("95% confidence intervals of dist are between: ", ci_d)


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
y7 = np.array(rolling_mean_reward)
y8 = np.array(rolling_mean_dist)


plt.plot(x1, y1, color="green")
plt.plot(x1, y7, color="blue")
plt.title("Reward-Episodes_DQN")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.savefig('Reward-Episodes_8000_DQN.png')
plt.close()
plt.plot(x2, y2, color="green")
plt.title("Minimum Reward-Episodes_DQN")
plt.xlabel("Episodes")
plt.ylabel("Minimum Reward")
plt.savefig('Minimum Reward-Episodes_8000_DQN.png')
plt.close()
plt.plot(x2, y3, color="green")
plt.title("Maximum Reward-Episodes_DQN")
plt.xlabel("Episodes")
plt.ylabel("Maximum Rewards")
plt.savefig('Maximum Reward-Episodes_8000_DQN.png')
plt.close()
plt.plot(x2, y4, color="green")
plt.title("Average Distance-Episodes_DQN")
plt.xlabel("Episodes")
plt.ylabel("Average Distance(m)")
plt.savefig('Average Distance-Episodes_8000_DQN.png')
plt.close()
plt.plot(x2, y5, color="green")
plt.title("Average Reward-Episodes_DQN")
plt.xlabel("Episodes")
plt.ylabel("Average Rewards")
plt.savefig('Average Reward-Episodes_8000_DQN.png')
plt.close()
plt.plot(x1, y6, color="green")
plt.plot(x1, y8, color="blue")
plt.title("Distance-Episodes_DQN")
plt.xlabel("Episodes")
plt.ylabel("Distance(m)")
plt.savefig('Distance-Episodes_8000_DQN.png')
plt.close()