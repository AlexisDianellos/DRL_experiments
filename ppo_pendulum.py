import time
import random
import numpy as np
import matplotlib.pylab as plt
plt.style.use('dark_background')
from tqdm.notebook import tqdm
from omegaconf import DictConfig

import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from IPython.display import Video

seed = 2023# the reason i choose to use a seed is that every time i restart my notebook my sequence of random actions that will be given to each of the 32 env's everytime we env.reset is the same
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

configs = {
    # experiment arguments
    "exp_name": "pendulum",
    "gym_id": "Pendulum-v1", # the id of from OpenAI gym
    # training arguments
    "learning_rate": 3e-4,#1e-3,  the learning rate of the optimizer
    "total_timesteps": 1000000, # total timesteps of the training
    "max_grad_norm": 0.5, # the maximum norm allowed for the gradient
    # PPO parameters
    "num_trajcts": 8, # N
    "max_trajects_length": 256, # T
    "gamma": 0.99, # gamma
    "gae_lambda":0.95, # lambda for the generalized advantage estimation
    "num_minibatches": 32, # number of mibibatches used in each gradient
    "update_epochs": 10, # number of full rollout storage creations
    "clip_epsilon": 0.2, # the surrogate clipping coefficient
    "ent_coef": 0.005, # entroy coefficient controlling the exploration factor C2
    "vf_coef": 0.5, # value function controlling value estimation importance C1
    # visualization and print parameters
    "num_returns_to_average": 3, # how many episodes to use for printing average return
    "num_episodes_to_average": 23, # how many episodes to use for smoothing of the return diagram
    }

# batch_size is the size of the flatten sequences when trajcts are flatten
configs['batch_size'] = int(configs['num_trajcts'] * configs['max_trajects_length'])
# number of samples used in each gradient
configs['minibatch_size'] = int(configs['batch_size'] // configs['num_minibatches'])

configs = DictConfig(configs)

run_name = f"{configs.gym_id}__{configs.exp_name}__{seed}__{int(time.time())}"

def make_env_func(gym_id, seed, idx, run_name, capture_video=False):
    def env_fun():
        # OPTIMIZATION: Only use render_mode for the specific env we are recording
        if capture_video and idx == 0:#only for env 0
            env = gym.make(gym_id, render_mode="rgb_array")#no pop up
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                episode_trigger=lambda x: x % 50 == 0#every 50 eps for idx 0 btw
            )
        else:
            # Standard env for pure training
            env = gym.make(gym_id)

        # Wrapper to track rewards (Critical for all_returns)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return env_fun

# Create Vector Envs
envs = []
for i in range(configs.num_trajcts):#0-31
    envs.append(make_env_func(configs.gym_id, seed + i, i, run_name))#each of the 32 env gets a diff seed - a diff starting point so that we can create Real Intelligence

envs = gym.vector.SyncVectorEnv(envs)

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, action_scale=2.0):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)#3 in -> 64
        self.fc2 = nn.Linear(64, 64)
        self.mean = nn.Linear(64, action_dim)#64-> 1=[-2,2]
        self.log_std = nn.Parameter(torch.ones(action_dim)*-0.5)#independant param, init w/ e^(1*(-0.5))=0.6
        self.max_action = action_scale

    def forward(self, x):
        x = torch.relu(self.fc1(x)) #[0,x]
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.mean(x)) * self.max_action
        std = torch.exp(self.log_std)
        return mean, std

class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)#3->64
        self.fc2 = nn.Linear(64, 64)
        self.v = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.v(x)

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = envs.single_observation_space.shape[0]
        action_dim = envs.single_action_space.shape[0]

        # Initialize the sub-modules
        self.actor = Actor(obs_dim, action_dim, action_scale=2.0)
        self.critic = Critic(obs_dim)

        # Apply orthogonal initialization to both
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def value_func(self, x):
        return self.critic(x)

    def policy(self, x, action=None):
        # Get mean and std from the Actor class
        mean, std = self.actor(x)

        # Create distribution
        probs = Normal(mean, std)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
    def gae(
    cur_observation,  # only latest state/observation from the env - 1curr state per env -> 32,4(bc for cartpole we have 4 observation per t)
    rewards,          # rewards collected from trajectories of shape [num_trajcts, max_trajects_length]32,64 -> 64 rewards of an episode for each env
    dones,            # binary marker of end of trajectories of shape [num_trajcts, max_trajects_length]32,64
    values,            # value estimates collected over trajectories of shape [num_trajcts, max_trajects_length]32,64
    cur_done,
    final_terminated_status
):
    advantages = torch.zeros((configs.num_trajcts, configs.max_trajects_length))#32,64
    last_advantage = 0

    # the value after the last step
    with torch.no_grad():
        last_value = agent.value_func(cur_observation).flatten()

    mask = 1.0 - final_terminated_status
    last_value = last_value * mask
    # reverse recursive to calculate advantages based on the delta formula
    for t in reversed(range(configs.max_trajects_length)): #reversed 0-63
        # mask if episode completed after step t
        mask = 1.0 - dones[:, t].float()#when an ep ends at t -> there is no next_value
        last_advantage = last_advantage * mask
        delta = rewards[:, t] + configs.gamma * last_value - values[:, t]#TD= raw surprise at t
        last_advantage = delta + configs.gamma * configs.gae_lambda * last_advantage#combines current surprise w/ surprise from all future steps
        advantages[:, t] = last_advantage
        last_value = values[:, t]

    advantages = advantages.to(device)
    returns = advantages + values #what we use to upd critic ((bc adv is the error + the prediction)) so we get the total correct answer

    return advantages, returns

def create_rollout(
    envs,            # parallel envs creating trajectories
    cur_observation, # starting observation of shape [num_trajcts, observation_dim] = 32,4 - each env gives us its first observation
    cur_done,        # current termination status of shape [num_trajcts,]
    all_returns      # a list to track returns
):
    """
    rollout phase: create parallel trajectories and store them in the rollout storage
    """

    # cache empty tensors to store the rollouts
    observations = torch.zeros((configs.num_trajcts, configs.max_trajects_length) +
                               envs.single_observation_space.shape).to(device)#32env,64steps,1 for pendulum
    actions = torch.zeros((configs.num_trajcts, configs.max_trajects_length) +
                          envs.single_action_space.shape).to(device)#32env,64steps,1action
    logprobs = torch.zeros((configs.num_trajcts, configs.max_trajects_length)).to(device)#32env,64steps
    rewards = torch.zeros((configs.num_trajcts, configs.max_trajects_length)).to(device)#32env,64steps
    dones = torch.zeros((configs.num_trajcts, configs.max_trajects_length)).to(device)#32env,64steps - only termination no truncation bc of pendulum
    values = torch.zeros((configs.num_trajcts, configs.max_trajects_length)).to(device)#32env,64steps

    final_terminated_status = torch.zeros(configs.num_trajcts).to(device)
    for t in range(configs.max_trajects_length):#64
        observations[:,t] = cur_observation#for all envs in observations select a time step t

        # give observation to the model and collect action, logprobs of actions, entropy and value
        with torch.no_grad():
            action, logprob, entropy, value = agent.policy(cur_observation)
        values[:,t] = value.flatten()
        actions[:,t] = action
        logprobs[:,t] = logprob.flatten()

        # apply the action to the env and collect observation and reward
        next_observation, reward, terminated, truncated, info = envs.step(action.cpu().numpy())

        rewards[:,t] = torch.tensor(reward).to(device).view(-1)
        dones[:,t] = torch.tensor(terminated).to(device)

        cur_observation = torch.Tensor(next_observation).to(device)
        cur_done = torch.tensor(np.logical_or(terminated, truncated)).to(device)

        if t == configs.max_trajects_length - 1:
            final_terminated_status = torch.tensor(terminated).float().to(device)

        # if an episode ended store its total reward for progress report
        if "episode" in info and "_episode" in info:
                    mask = info["_episode"]
                    episode_rewards = info["episode"]["r"]

                    # Iterate through the batch to find finished episodes
                    for i, is_finished in enumerate(mask):
                        if is_finished:
                            # Append the total reward (r) for this specific environment
                            all_returns.append(episode_rewards[i])

    # create the rollout storage
    rollout = {
        'cur_observation': cur_observation,
        'cur_done': cur_done, # term/trunv of state just collected
        'final_terminated_status': final_terminated_status,
        'observations': observations,
        'actions': actions,
        'logprobs': logprobs,
        'values': values,
        'dones': dones, # historical record of termination only of entire batch
        'rewards': rewards
    }

    return rollout

class Storage(Dataset):# each storage block is per 64t for all 32 envs combined
    def __init__(self, rollout, advantages, returns, envs):
        # fill in the storage and flatten the parallel trajectories
        self.observations = rollout['observations'].reshape((-1,) + envs.single_observation_space.shape)
        self.logprobs = rollout['logprobs'].reshape(-1)
        self.actions = rollout['actions'].reshape((-1,) + envs.single_action_space.shape)
        self.advantages = advantages.reshape(-1)
        self.returns = returns.reshape(-1)

    def __getitem__(self, ix: int):
        item = [
            self.observations[ix],
            self.logprobs[ix],
            self.actions[ix],
            self.advantages[ix],
            self.returns[ix]
        ]
        return item

    def __len__(self) -> int:
        return len(self.observations)
    
    def loss_clip(
    mb_oldlogporb,     # old logprob of mini batch actions collected during the rollout
    mb_newlogprob,     # new logprob of mini batch actions created by the new policy
    mb_advantages      # mini batch of advantages collected during the the rollout
):
    """
    policy loss with clipping to control gradients
    """
    ratio = torch.exp(mb_newlogprob - mb_oldlogporb)
    policy_loss = -mb_advantages * ratio
    # clipped policy gradient loss enforces closeness
    clipped_loss = -mb_advantages * torch.clamp(ratio, 1 - configs.clip_epsilon, 1 + configs.clip_epsilon)
    pessimistic_loss = torch.max(policy_loss, clipped_loss).mean()
    return pessimistic_loss


def loss_vf(
    mb_oldreturns,  # mini batch of old returns collected during the rollout
    mb_newvalues    # minibach of values calculated by the new value function
):
    """
    enforcing the value function to give more accurate estimates of returns
    """
    mb_newvalues = mb_newvalues.view(-1)
    loss = 0.5 * ((mb_newvalues - mb_oldreturns) ** 2).mean()
    return loss

agent = Agent(envs).to(device)

optimizer = optim.Adam(agent.parameters(), lr=configs.learning_rate)

# track returns
all_returns = []
debug_printed = False

# initialize the game
cur_observation, _ = envs.reset(seed=seed)
cur_observation = torch.Tensor(cur_observation).to(device)
cur_done = torch.zeros(configs.num_trajcts).to(device)

# progress bar
num_updates = configs.total_timesteps // configs.batch_size
progress_bar = tqdm(total=num_updates)

for update in range(1, num_updates + 1):

    ##############################################
    # Phase 1: rollout creation

    # parallel envs creating trajectories
    rollout = create_rollout(envs, cur_observation, cur_done, all_returns)

    cur_done = rollout['cur_done']
    cur_observation = rollout['cur_observation']
    rewards = rollout['rewards']
    dones = rollout['dones']
    values = rollout['values']
    final_terminated_status = rollout['final_terminated_status']

    # --- DEBUGGING BLOCK ---
    # If all_returns is still empty after 20 updates, force a check
    if len(all_returns) == 0 and update == 20:
        print("DEBUG: 20 updates passed and no returns. Checking a random step info...")
        # We can't easily check 'info' here as it's inside create_rollout
        # But the create_rollout update below fixes the collection logic.

    # calculating advantages
    advantages, returns = gae(cur_observation, rewards, dones, values, cur_done,final_terminated_status)

    # a dataset containing the rollouts
    dataset = Storage(rollout, advantages, returns, envs)

    # a standard dataloader made out of current storage
    trainloader = DataLoader(dataset, batch_size=configs.minibatch_size, shuffle=True)


    ##############################################
    # Phase 2: model update

    # linearly shrink the lr from the initial lr to zero
    frac = 1.0 - (update - 1.0) / num_updates
    optimizer.param_groups[0]["lr"] = frac * configs.learning_rate

    # training loop
    for epoch in range(configs.update_epochs):
        for batch in trainloader:
            mb_observations, mb_logprobs, mb_actions, mb_advantages, mb_returns = batch

            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # we calculate the distribution of actions through the updated model revisiting the old trajectories
            _, mb_newlogprob, mb_entropy, mb_newvalues = agent.policy(mb_observations, mb_actions)

            policy_loss = loss_clip(mb_logprobs, mb_newlogprob, mb_advantages)

            value_loss = loss_vf(mb_returns, mb_newvalues)

            # average entory of the action space
            entropy_loss = mb_entropy.mean()

            # full weighted loss
            loss = policy_loss - configs.ent_coef * entropy_loss + configs.vf_coef * value_loss

            optimizer.zero_grad()
            loss.backward()

            # extra clipping of the gradients to avoid overshoots
            nn.utils.clip_grad_norm_(agent.parameters(), configs.max_grad_norm)
            optimizer.step()

    # progress bar
    if len(all_returns) > configs.num_returns_to_average:
        progress_bar.set_description(f"episode return: {np.mean(all_returns[-configs.num_returns_to_average:]):.2f}")
        progress_bar.update()

envs.close()