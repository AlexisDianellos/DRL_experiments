#q learning w/ q table
import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt

def run(episodes,render=False,is_training=True):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human' if render else None )

    if is_training:
      q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
    else:
      f = open('frozen_lake8x8_act.pkl','rb')
      q = pickle.load(f)
      f.close()
    
    learning_rate_a = 0.9 # hyperparams the q learning formula depends on
    discount_factor_g = 0.9

    #epsilon greedy algo init
    epsilon = 1 # 1 = 100% random actions
    epsilon_decay_rate = 0.00005 #to decay epsilon for less randomness - 0.0001 means every 10k episodes epsilon will be 0
    rng = np.random.default_rng() #random num gen

    rewards_per_episode = np.zeros(episodes)
    
    for i in range(episodes):
      state=env.reset()[0] #states 0-63
      terminated=False
      truncated=False

      while(not terminated and not truncated):
          #0left 1 down 2 right 3 up
          if is_training and rng.random()<epsilon:  
            action=env.action_space.sample()#exploration
          else:
            action = np.argmax(q[state,:])#exploitation

          new_state,reward,terminated,truncated,_=env.step(action)

          if terminated or truncated:
            target=reward# bc if we get truncated or term there is no meaningful new_state
          else:
            target=reward + discount_factor_g * np.max(q[new_state,:])

          if is_training:
            q[state,action] = q[state,action] + learning_rate_a * (target - q[state,action])
          #formula updates w reward that it takes for the step also largest q value in new state

          state=new_state
    
      epsilon = max(epsilon - epsilon_decay_rate, 0)

      if(epsilon==0):
        learning_rate_a = 0.0001#were no longer exploring

      if reward == 1:
        rewards_per_episode[i] = 1
        
    env.close()

    window_size = 100
    sum_rewards = np.convolve(rewards_per_episode, np.ones(window_size), mode='full')[:episodes]
    plt.figure(figsize=(10,5))
    plt.plot(sum_rewards)
    plt.xlabel("Episode")
    plt.ylabel(f"Sum of rewards per last {window_size} episodes")
    plt.title("Frozen Lake 8x8 Q-learning")
    plt.grid(True)
    plt.savefig('frozen_lake8x8.png')
    plt.close()

    if is_training:
      f = open("frozen_lake4x4_newtest2.pkl","wb")
      pickle.dump(q, f)
      f.close()
  
if __name__ == '__main__':
    # run(15000)

    run(50000,render=False,is_training=True)