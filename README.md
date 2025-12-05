# Deep Reinforcement Learning & Safe Reinforcement Learning Experiments
As part of my multiagent DRL thesis I am going to be documenting experiments and implementation of algorithms in gymnasium env's

# Q-learning w/ Frozen Lake 4x4 and 8x8 environment Slippery OFF
4x4, 15.000 episodes of training, epsilon_decay_rate = 0.0001
<img width="1000" height="500" alt="frozen_lake4x4" src="https://github.com/user-attachments/assets/ecab3aa9-d721-4f80-bfec-891b6ec0d71a" />
Outcome: expected performance at around 10.000 episodes the algorithm stops experiments and starts following the q table and has learned a policy to reach the reward every time
- However considering the simplicity of the 4x4 frozen lake map there had to be a quicker way requiring much less training

4x4 15.000 episodes of training, epsilon_decay_rate = 0.0005
<img width="1000" height="500" alt="frozen_lake4x4_number2" src="https://github.com/user-attachments/assets/eec8d6e6-916b-4387-ad1f-38000b3b5d20" />
Outcome: with the increased epsilon_decay_rate = 0.0005, epsilon reaches zero much quicker at around 1.5k episodes. When the algorithm starts following the q table post 1.5k episodes it is obvious that it is able to reach the reward each time
- 15000 episodes was overkill though

8x8 100.000 episodes of training, epsilon_decay_rate = 0.00001
<img width="1000" height="500" alt="frozen_lake8x8" src="https://github.com/user-attachments/assets/e1c75a9c-5e6e-40ae-9ee3-021610559014" />
- 100.000 episodes was overkill though

8x8 50.000 episodes of training, epsilon_decay_rate = 0.00005
<img width="1000" height="500" alt="frozen_lake8x8_newtest2" src="https://github.com/user-attachments/assets/b2d53af4-79c7-4905-9255-7b7ae89c9f1b" />
- 50.000 episodes was overkill though
- I tried with 10.000 episodes but the model never reached the reward due to the size of the map
- The ideal is around 25.000 for training in the 8x8 map with q learning

# DEEP-Q-learning w/ Frozen Lake 4x4 environment Slippery OFF/ON
4x4, 1000 episodes with slipper off
<br/>
<img width="640" height="480" alt="frozen_lake_dql" src="https://github.com/user-attachments/assets/3519c34b-0ecf-48c3-9e71-39041dad74a4" />
- only reaches 80percent episode success but is exepected w/ the number of training episodes

4x4 10000 episodes with slippery on
<br/>
<img width="640" height="480" alt="frozen_lake_dql_4x4_slippery" src="https://github.com/user-attachments/assets/02a24853-0a0d-4e32-9a49-38d39aa793ec" />
- 10.000 episodes overkill i found that 3000 episode is the most effective however due to the slippery parameters randomness the episode success never reaches more than 60 percent
- the answer is not more training episodes though

# PPO (Generalised Advantage Estimation) w/ Cartpole-v1 (Discrete Environment)
<img width="850" height="470" alt="ppo_crapole" src="https://github.com/user-attachments/assets/f01ce3ed-cc2c-4617-8453-6655b8b19c84" />
<br/>
- I used 32 parallel enviroments / gae for advantage estimation / single NN architecture with 2 heads <br/>
- I played around a lot with gamma but found a gamma=0.99 and gae_lambda=0.95 to be best <br/>
- I was very surprised to see how quickly it was able to reach the max 500 reward on cartpole (only about 5 updates in) <br/>
- The drops in reward ensure the implementation is actually correct -> still Explores despite reach max reward <br/>
- Video : https://github.com/user-attachments/assets/eaa8501b-f1de-462e-a58d-96fa0e4c0a35

# PPO (Generalised Advantage Estimation) w/ Pendulum-v1 (Continuous Environment)
<img width="1190" height="590" alt="ppo_pendulum1" src="https://github.com/user-attachments/assets/78673cca-56c1-47b0-a91f-87bc28265fd4" />
- Upon attempting the same architecture as the cartpole enviroment I ran into horrible performance -> pendulum env is a lot more complex model has to figure out physics
<img width="1189" height="590" alt="ppo_pendulum_200k_ts" src="https://github.com/user-attachments/assets/c6f74bde-5bc7-4338-85f4-d8d57ada1ed2" />
- Lowered learning rate to 3e-4: Continuous action spaces are highly sensitive, so a high LR can push the π into a region where it cannot recover <br/>
- Increased the trajectories of the agents env interaction before training (max_trajcts_length): Pendulum is a physics problem and the short traj would prevent the agent from understanding the benefit of momentum <br/>
- Reduced the entropy coefficient: was preventing the agent from exploiting <br/>
<img width="1189" height="590" alt="pednulu_ppo_good" src="https://github.com/user-attachments/assets/049eaf37-c1c6-4fbf-9d4c-dcd31fd8bd2c" />
- However despite a slight performance improvement something was missing - the agent couldnt understand the benefit of momentum <br/>
  So:<br/>
    - Instead of one neural net splitting into two heads, I implemented two distinct networks: one for the Actor (Policy) and one for the Critic: Clear that the gradient updates from the Critic where interfering w/ Actor<br/>
    - Opted for a simpler 2 layer MLP instead of complex residual blocks and layernorm: for increased training speed<br/>

  Its clear the model was able to achieve perfect performance
- Video : https://github.com/user-attachments/assets/2b303bd8-6c1d-49c2-9128-de3ce9f20c9c

# TODO:
- Dual Gradient Descent
- PPO La grandian Paper enviroment implementation

