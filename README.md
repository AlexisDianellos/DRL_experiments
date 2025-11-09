# DRL_experiments
As part of my multiagent DRL thesis I am going to be documenting experiments and implementation of algorithms in gymnasium env's

# Q-learning w/ Frozen Lake 4x4 and 8x8 environment
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
