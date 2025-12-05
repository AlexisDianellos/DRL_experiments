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
    "exp_name": "cartpole",
    "gym_id": "CartPole-v1", # the id of from OpenAI gym
    # training arguments
    "learning_rate": 1e-3, # the learning rate of the optimizer
    "total_timesteps": 1000000, # total timesteps of the training
    "max_grad_norm": 0.5, # the maximum norm allowed for the gradient
    # PPO parameters
    "num_trajcts": 32, # N
    "max_trajects_length": 64, # T
    "gamma": 0.99, # gamma
    "gae_lambda":0.95, # lambda for the generalized advantage estimation
    "num_minibatches": 2, # number of mibibatches used in each gradient
    "update_epochs": 2, # number of full rollout storage creations
    "clip_epsilon": 0.2, # the surrogate clipping coefficient
    "ent_coef": 0.01, # entroy coefficient controlling the exploration factor C2
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

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return env_fun

# Create Vector Envs
envs = []
for i in range(configs.num_trajcts):#0-31
    envs.append(make_env_func(configs.gym_id, seed + i, i, run_name))#each of the 32 env gets a diff seed - a diff starting point so that we can create Real Intelligence

envs = gym.vector.SyncVectorEnv(envs)

class FCBlock(nn.Module):#residual fc block
#its goal is to take the raw env state and transform it into a richer representation that is easier for the 2 heads to interpret
    def __init__(self, embd_dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(embd_dim),#32envsx64
            #normalizes each feature - prevents 1 feature from dominating
            # ex: state=[1.0,-2.0,0.5,3.0] -> normalized=[0.214,-1.523,-0.072,1.382]

            nn.GELU(),#max(0,x) but pos and neg values are scaled
            # ex: normalized=[0.214,-1.523,-0.072,1.382] -> GELU=[0.13,-0.07,-0.034,1.237]

            nn.Linear(embd_dim, 4*embd_dim),#64->256 - thinking step allowing the block to combine the features in diff ways
            nn.GELU(),
            nn.Linear(4*embd_dim, embd_dim),#256->64 - compress the features
            nn.Dropout(dropout)#randomly drops(zeros) 1/4 features, depending on dropout_rate=0.2 - 20percent of features for each of 32envs 0 preventing reliance on just a few strong signals
        )
    def forward(self, x):
        return x + self.block(x) # original input added to output -> residual connection


class Agent(nn.Module):
    """an agent that creates actions and estimates values"""
    def __init__(self, env_observation_dim, action_space_dim, embd_dim=64, num_blocks=2):#4,2,64,2
        super().__init__()
        self.embedding_layer = nn.Linear(env_observation_dim, embd_dim)#4->64
        self.shared_layers = nn.Sequential(*[FCBlock(embd_dim=embd_dim) for _ in range(num_blocks)])#sequential means the output of the first block goes into the second block
        self.value_head = nn.Linear(embd_dim, 1)#64->1 V
        self.policy_head = nn.Linear(embd_dim, action_space_dim)#64-> 2 Action
        # orthogonal initialization with a hi entropy for more exploration at the start
        torch.nn.init.orthogonal_(self.policy_head.weight, 0.01)#weights init

    def value_func(self, state):
        hidden = self.shared_layers(self.embedding_layer(state))
        value = self.value_head(hidden)
        return value

    def policy(self, state, action=None):
        hidden = self.shared_layers(self.embedding_layer(state))
        logits = self.policy_head(hidden)
        # PyTorch categorical class helpful for sampling and probability calculations
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.value_head(hidden)
    
    def gae(
    cur_observation,  # only latest state/observation from the env - 1curr state per env -> 32,4(bc for cartpole we have 4 observation per t)
    rewards,          # rewards collected from trajectories of shape [num_trajcts, max_trajects_length]32,64 -> 64 rewards of an episode for each env
    dones,            # binary marker of end of trajectories of shape [num_trajcts, max_trajects_length]32,64
    values            # value estimates collected over trajectories of shape [num_trajcts, max_trajects_length]32,64
):
    advantages = torch.zeros((configs.num_trajcts, configs.max_trajects_length))#32,64
    last_advantage = 0

    # the value after the last step
    with torch.no_grad():
        last_value = agent.value_func(cur_observation).reshape(1, -1)

    # reverse recursive to calculate advantages based on the delta formula
    for t in reversed(range(configs.max_trajects_length)): #reversed 0-63
        # mask if episode completed after step t
        mask = 1.0 - dones[:, t]#when an ep ends at t -> there is no next_value
        last_value = last_value * mask
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
                               envs.single_observation_space.shape).to(device)#32env,64steps,4features
    actions = torch.zeros((configs.num_trajcts, configs.max_trajects_length) +
                          envs.single_action_space.shape).to(device)#32env,64steps,1action
    logprobs = torch.zeros((configs.num_trajcts, configs.max_trajects_length)).to(device)#32env,64steps
    rewards = torch.zeros((configs.num_trajcts, configs.max_trajects_length)).to(device)#32env,64steps
    dones = torch.zeros((configs.num_trajcts, configs.max_trajects_length)).to(device)#32env,64steps
    values = torch.zeros((configs.num_trajcts, configs.max_trajects_length)).to(device)#32env,64steps

    for t in range(configs.max_trajects_length):#64
        observations[:,t] = cur_observation#for all envs in observations select a time step t
        dones[:,t] = cur_done

        # give observation to the model and collect action, logprobs of actions, entropy and value
        with torch.no_grad():#ensures we dont upd weights
            action, logprob, entropy, value = agent.policy(cur_observation)
        values[:,t] = value.flatten()
        actions[:,t] = action
        logprobs[:,t] = logprob

        # apply the action to the env and collect observation and reward
        cur_observation, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
        cur_done = np.logical_or(terminated, truncated)

        rewards[:,t] = torch.tensor(reward).to(device).view(-1)
        cur_observation = torch.Tensor(cur_observation).to(device)
        cur_done = torch.Tensor(cur_done).to(device)

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
        'cur_done': cur_done,
        'observations': observations,
        'actions': actions,
        'logprobs': logprobs,
        'values': values,
        'dones': dones,
        'rewards': rewards
    } #packages these tensors collected over 64 steps into a dictionary for the next phase -> gae

    return rollout

class Storage(Dataset):# each storage block is per 64t for all 32 envs combined
    def __init__(self, rollout, advantages, returns, envs):
        # fill in the storage and flatten the parallel trajectories -> so 32envx64t = 2048 values
        self.observations = rollout['observations'].reshape((-1,) + envs.single_observation_space.shape)
        self.logprobs = rollout['logprobs'].reshape(-1)
        self.actions = rollout['actions'].reshape((-1,) + envs.single_action_space.shape).long()
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

agent = Agent(
    env_observation_dim=envs.single_observation_space.shape[0],
    action_space_dim=envs.single_action_space.n
).to(device)

optimizer = optim.Adam(agent.parameters(), lr=configs.learning_rate)

# track returns
all_returns = []
debug_printed = False

# initialize the game
cur_observation, _ = envs.reset(seed=seed)
cur_observation = torch.Tensor(cur_observation).to(device)
cur_done = torch.zeros(configs.num_trajcts).to(device)

# progress bar
num_updates = configs.total_timesteps // configs.batch_size #1mil/2048batches is around 488
progress_bar = tqdm(total=num_updates)

for update in range(1, num_updates + 1):#total times an agent must collect a batch,grade it,update net

    ##############################################
    # Phase 1: rollout creation

    # parallel envs creating trajectories
    rollout = create_rollout(envs, cur_observation, cur_done, all_returns)

    cur_done = rollout['cur_done']
    cur_observation = rollout['cur_observation']
    rewards = rollout['rewards']
    dones = rollout['dones']
    values = rollout['values']

    #debug
    if len(all_returns) == 0 and update == 20:
        print("DEBUG: 20 updates passed and no returns. Checking a random step info...")

    # calculating advantages
    advantages, returns = gae(cur_observation, rewards, dones, values)

    # a dataset containing the rollouts
    dataset = Storage(rollout, advantages, returns, envs)

    # a standard dataloader made out of current storage
    trainloader = DataLoader(dataset, batch_size=configs.minibatch_size, shuffle=True)


    ##############################################
    # Phase 2: model update

    # linearly shrink the lr from the initial lr to 0
    frac = 1.0 - (update - 1.0) / num_updates
    optimizer.param_groups[0]["lr"] = frac * configs.learning_rate

    # training loop
    for epoch in range(configs.update_epochs):#how many times the 2048 data points are fed into the net before collecting new data - 10
        for batch in trainloader:
            mb_observations, mb_logprobs, mb_actions, mb_advantages, mb_returns = batch

            # runs collected obs through net for new action probabilities and v
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