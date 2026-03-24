from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
try:
    from gymnasium.wrappers import FrameStackObservation
    HAS_FRAMESTACK_OBS = True
except ImportError:
    from gymnasium.wrappers import FrameStack
    HAS_FRAMESTACK_OBS = False

# -------------------------
# 0) Device + GPU sanity
# -------------------------
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print("✅ CUDA:", torch.cuda.get_device_name(0))
        return d
    d = torch.device("cpu")
    print("⚠️ CUDA not available. Using CPU.")
    return d
'''
observation space: Box(0, 255, (96, 96, 3), uint8)
where (0-255)range -> we have a tuple where 96x96x3
imagine each pixel: has 3 numbers -> pixel1 (0,0) -> [5,100,200]
                                     pixel2 (0,1) -> [10,10,10]
                                     pixel3 (0,2) -> [20,20,20]
'''
# -------------------------
# 1) Hyperparameters
# -------------------------
num_envs        = 8
horizon         = 128            # steps per env before ppo update (rollout length)
gamma           = 0.99
gae_lambda      = 0.95

# PPO update
ppo_epochs      = 4              # how many times the same data is used to update the network (which data is 8env x horizon 128)
minibatch_size  = 256            # B = num_envs*horizon = 1024 here -> 4 minibatches
clip_coef       = 0.2

# Loss weights
vf_coef         = 0.5
ent_coef        = 0.005         # exploration; if too random, try 0.005 or 0.001

# Optimizer
lr              = 1e-4
max_grad_norm   = 0.5
frame_stack = 4

# Training length
num_updates     = 3000  # the entire process of 1 collect rollout, 2 compute adv,returns 3 train policy for epochs is (3 is num_updates -> how many times we do this update)

SAVE_PATH = "ppo_carracing3.pt"
HISTORY_PATH = "ppo_history3.npz"

def make_env(seed: int, idx: int, capture_video: bool = False, run_name: str = "ppo_carracing", stack_size: int = 4):
    def thunk():
        render_mode = "rgb_array" if (capture_video and idx == 0) else None

        env = gym.make(
            "CarRacing-v3",
            continuous=True,
            render_mode=render_mode,
        )

        # 4-frame stack
        if HAS_FRAMESTACK_OBS:
            env = FrameStackObservation(env, stack_size)
        else:
            env = FrameStack(env, num_stack=stack_size)

        env = gym.wrappers.RecordEpisodeStatistics(env)

        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=f"videos/{run_name}",
                episode_trigger=lambda ep: ep % 50 == 0,
                disable_logger=True,
            )

        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)

        return env

    return thunk

def make_vec_env(num_envs: int, seed: int, capture_video: bool = False, run_name: str = "ppo_carracing", stack_size: int = 4):
    return gym.vector.AsyncVectorEnv(
        [
            make_env(
                seed=seed,
                idx=i,
                capture_video=capture_video,
                run_name=run_name,
                stack_size=stack_size,
            )
            for i in range(num_envs)
        ]
    )

def preprocess_obs(obs, device):
    """
    Supports:
      single env, no stack:      (96, 96, 3)
      vector env, no stack:      (N, 96, 96, 3)
      single env, 4-stack:       (4, 96, 96, 3)
      vector env, 4-stack:       (N, 4, 96, 96, 3)

    Returns:
      no stack:                  (N, 3, 96, 96)
      4-stack:                   (N, 12, 96, 96)
    """
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs)
    else:
        obs = torch.as_tensor(np.array(obs))

    # Single frame, single env: (96,96,3) -> (1,96,96,3)
    if obs.ndim == 3:
        obs = obs.unsqueeze(0)

    # Single env, stacked: (K,96,96,3) -> (1,K,96,96,3)
    if obs.ndim == 4 and obs.shape[-1] == 3 and obs.shape[0] == frame_stack:
        obs = obs.unsqueeze(0)

    obs = obs.to(device=device, dtype=torch.float32) / 255.0

    # Non-stacked: (N,H,W,C) -> (N,C,H,W)
    if obs.ndim == 4:
        obs = obs.permute(0, 3, 1, 2).contiguous()
        return obs

    # Stacked: (N,K,H,W,C) -> (N,K,C,H,W) -> (N,K*C,H,W)
    if obs.ndim == 5:
        obs = obs.permute(0, 1, 4, 2, 3).contiguous()   # (N,K,C,H,W)
        N, K, C, H, W = obs.shape
        obs = obs.reshape(N, K * C, H, W)               # (N,12,H,W)
        return obs

    raise ValueError(f"Unexpected obs shape: {tuple(obs.shape)}")

class ActorCriticCNN(nn.Module):
    def __init__(self, obs_channels: int = 12, action_dim: int = 3):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Conv2d(obs_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
        )

        self.policy_mean = nn.Linear(512, action_dim)
        self.value_head = nn.Linear(512, 1)

        # learned log std, one per action dim
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        hidden = self.trunk(obs)
        mean = self.policy_mean(hidden)          # (N, action_dim)
        value = self.value_head(hidden).squeeze(-1)
        return mean, value

    def act(self, obs):
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)

        raw_action = dist.sample()
        logprob = dist.log_prob(raw_action).sum(-1)
        entropy = dist.entropy().sum(-1)

        action = raw_action.clone()
        action[:, 0] = torch.tanh(action[:, 0])     # steering [-1,1]
        action[:, 1] = torch.sigmoid(action[:, 1])  # gas [0,1]
        action[:, 2] = torch.sigmoid(action[:, 2])  # brake [0,1]

        return action, raw_action, logprob, entropy, value

    def evaluate_actions(self, obs, raw_actions):
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)

        logprob = dist.log_prob(raw_actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logprob, entropy, value

@torch.no_grad()
def collect_rollout(envs, model, device, horizon, next_obs):
    """
    Collect T=horizon steps from vector envs.

    Inputs
    ------
    envs: gymnasium.vector.VectorEnv
    model: ActorCriticCNN
    device: torch device
    horizon: int (T)
    next_obs: latest raw obs from envs.reset() or previous call
              shape: (N, 96, 96, 3), dtype uint8

    Returns
    -------
    batch: dict of tensors with shapes:
        obs:      (T, N, 3, 96, 96)   float32 in [0,1]   (preprocessed)
        actions:  (T, N)              int64
        logprobs: (T, N)              float32
        rewards:  (T, N)              float32
        dones:    (T, N)              float32 (1.0 if done else 0.0)  done = terminated OR truncated
        values:   (T, N)              float32
    next_obs: raw obs after last step
        shape: (N, 96, 96, 3), uint8
    last_done: (N,) float32
    last_value:(N,) float32  value estimate for next_obs (used for GAE bootstrap)
    infos: last info dict returned by env.step (useful for logging episodes)
    """
    # N = number of parallel envs
    N = envs.num_envs

    # Allocate storage (on device for simplicity)
    obs_buf      = torch.zeros((horizon,N,frame_stack* 3, 96, 96), device=device, dtype=torch.float32)
    actions_buf  = torch.zeros((horizon, N, 3),            device=device, dtype=torch.float32)
    raw_actions_buf  = torch.zeros((horizon, N, 3), device=device, dtype=torch.float32)
    logprob_buf  = torch.zeros((horizon, N),            device=device, dtype=torch.float32)
    rewards_buf  = torch.zeros((horizon, N),            device=device, dtype=torch.float32)
    dones_buf    = torch.zeros((horizon, N),            device=device, dtype=torch.float32)
    values_buf   = torch.zeros((horizon, N),            device=device, dtype=torch.float32)

    infos = None

    for t in range(horizon):
        # next_obs raw:
        #   before preprocess: (N, 96, 96, 3) uint8
        obs_t = preprocess_obs(next_obs, device)   # after: (N, 3, 96, 96) float32

        # Store obs used for action at time t
        obs_buf[t] = obs_t

        # Sample action from current policy
        action, raw_action, logprob, entropy, value = model.act(obs_t)
        actions_buf[t] = action
        raw_actions_buf[t] = raw_action#entropy=how unsure the model is of taking action - entropy adds bonus to encourage exporing other actions

        # action: (N,) int64 ; logprob: (N,) ; value: (N,)

        actions_buf[t] = action
        logprob_buf[t] = logprob
        values_buf[t]  = value

        # Step envs (Gymnasium vector env expects actions as numpy-ish)
        next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())

        # reward: (N,) float
        # terminated/truncated: (N,) bool
        done = terminated | truncated

        rewards_buf[t] = torch.as_tensor(reward, device=device, dtype=torch.float32)
        dones_buf[t]   = torch.as_tensor(done,   device=device, dtype=torch.float32)
    # Bootstrap value for the final next_obs
    obs_last = preprocess_obs(next_obs, device)     # (N, 3, 96, 96)
    _, last_value = model.forward(obs_last)         # last_value: (N,)

    last_done = dones_buf[-1]                       # (N,) float32

    batch = {
        "obs": obs_buf,
        "actions": actions_buf,
        "raw_actions": raw_actions_buf,
        "logprobs": logprob_buf,
        "rewards": rewards_buf,
        "dones": dones_buf,
        "values": values_buf,
    }

    return batch, next_obs, last_done, last_value, infos

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """
    Inputs (from rollout):
      rewards:    (T, N) float32
      values:     (T, N) float32   V(s_t)
      dones:      (T, N) float32   1.0 if done else 0.0  (done = terminated OR truncated)
      last_value: (N,)   float32   V(s_T) for bootstrap (from final next_obs)

    Returns:
      advantages: (T, N) float32
      returns:    (T, N) float32   = advantages + values
    """
    T, N = rewards.shape
    advantages = torch.zeros((T, N), device=rewards.device, dtype=torch.float32)

    # running advantage estimate (per env)
    gae = torch.zeros((N,), device=rewards.device, dtype=torch.float32)

    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]# make sure t is not terminal state bc we cant take action from it if it is
        
        # Value of the next state
        next_value = last_value if t == T - 1 else values[t + 1]
        
        # D = Reward + discounted how good is next state - how good is this s
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]

        gae = delta + gamma * lam * nonterminal * gae# so this increases as we go from T to 0 bc we first adv[0] is the biggest

        advantages[t] = gae # A(s,a)
    
    # So A(s,a)=Q(s,a)-V(s)
    # A(s,a)-> how right/wrong prediction after seeing what happened
    # V(s)-> nn prediction

    # So Q(s,a)=A(s,a)+V(s), where Qreturns=(Q-V)+V so returns=Q
    
    returns = advantages + values  # returns is what i use to train the value function net
    return advantages, returns

def ppo_policy_loss(new_logprob, old_logprob, advantages, clip_coef: float):
    """
    Loss = min ( r(t)A(t), clip(r(t), 1-e, 1+e)A(t) ) , where r(t)=π(a|s)/πold(a|s)


    Inputs (all same shape, typically (B,)):
      new_logprob: log pi_theta(a|s)
      old_logprob: log pi_theta_old(a|s)  (from rollout buffer, detached)
      advantages:  advantage estimates (usually normalized)
      clip_coef:   epsilon, e.g. 0.2

    Returns:
      loss: scalar tensor (to minimize)
      approx_kl: scalar tensor (useful for logging / early stopping)
      clipfrac: scalar tensor (fraction of samples that got clipped)
    """
    # ratio = pi(a|s) / pi_old(a|s) in log space
    log_ratio = new_logprob - old_logprob
    ratio = torch.exp(log_ratio)

    # unclipped and clipped objectives
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages

    # LOSS
    loss = -torch.mean(torch.min(surr1, surr2))

    # Diagnostics (common PPO logging)
    approx_kl = torch.mean(ratio - 1.0 - log_ratio)  # approx KL(old || new)
    clipfrac = torch.mean((torch.abs(ratio - 1.0) > clip_coef).float())

    return loss, approx_kl, clipfrac

def value_loss(values_pred, returns):
    # values_pred, returns: (B,)
    return 0.5 * F.mse_loss(values_pred, returns)

def entropy_bonus(entropy, ent_coef):
    """
    Entropy bonus term for PPO.

    Inputs
    ------
    entropy:  (B,) tensor
        Per-sample action distribution entropy, usually from dist.entropy().
        For discrete actions (Categorical), higher entropy = more random / exploratory policy.
    ent_coef: float
        Coefficient controlling how strongly we encourage exploration.

    Returns
    -------
    bonus: scalar tensor
        A scalar you ADD to the objective (or equivalently SUBTRACT from the loss).
    """
    return ent_coef * entropy.mean()

def train_step(
    model,
    optimizer,
    batch,
    *,
    ppo_epochs: int = 4,
    minibatch_size: int = 256,
    clip_coef: float = 0.2,#used in the ppo loss clipping
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,# if its 0 no exploration
    max_grad_norm: float = 0.5,
):
    """
    At this point in PPO I have:
      run agent in parallel env and stored what happened
      compute advantages
      compute returns
    That is a Batch,
    Perform PPO updates on one rollout batch.

    Expected batch dict (from your collect_rollout + GAE):
      batch["obs"]:       (T, N, 3, 96, 96)   float32, where T is num of steps per env
      batch["actions"]:   (T, N)              int64
      batch["logprobs"]:  (T, N)              float32   (old logprob)
      batch["values"]:    (T, N)              float32   (old value) [optional, for vf clipping/logging]
      batch["advantages"]:(T, N)              float32
      batch["returns"]:   (T, N)              float32

    This function flattens (T, N) -> (B,) where B = T*N, then runs PPO epochs/minibatches.
    """
    device = batch["obs"].device
    T, N = batch["logprobs"].shape
    B = T * N

    obs = batch["obs"].reshape(B, frame_stack * 3, 96, 96)
    raw_actions = batch["raw_actions"].reshape(B, 3).detach()
    old_logprobs = batch["logprobs"].reshape(B).detach()
    advantages = batch["advantages"].reshape(B).detach()
    returns = batch["returns"].reshape(B).detach()

    # Advantage normalization (very common in PPO) - training more stable
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Logging accumulators
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0
    total_clipfrac = 0.0
    num_updates = 0

    # Indices for shuffling
    idxs = np.arange(B)#list of all idxs from 0 to B-1

    for _ in range(ppo_epochs):
        np.random.shuffle(idxs)# random order of each minibatch (minibatch )

        for start in range(0, B, minibatch_size): #B/minibatch_size = num of range
            mb_idx = idxs[start:start + minibatch_size]
            mb_idx = torch.as_tensor(mb_idx, device=device, dtype=torch.long)

            #pull out of this minibatch: obs,act,oldlogporb,adv,returns
            mb_obs = obs[mb_idx]                 # (mb, 3, 96, 96)
            mb_raw_actions = raw_actions[mb_idx]        # (mb,)
            mb_old_logp = old_logprobs[mb_idx]   # (mb,)
            mb_adv = advantages[mb_idx]          # (mb,)
            mb_returns = returns[mb_idx]         # (mb,)

            # Rerun model on minibatch
            new_logp, entropy, values_pred = model.evaluate_actions(mb_obs, mb_raw_actions)

            policy_loss, approx_kl, clipfrac = ppo_policy_loss(
                new_logp, mb_old_logp, mb_adv, clip_coef
            )

            v_loss = 0.5 * F.mse_loss(values_pred, mb_returns)

            # --- Entropy bonus ---
            ent = entropy.mean()
            # ent helps the network want to explore more

            # Total PPO loss (minimize) -> that then goes an affects each gradient
            loss = policy_loss + vf_coef * v_loss - ent_coef * ent

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # gradients are the parametres that get computed in every backward pass and they can explode in size SO we clip them without changing their direction (positive remains pos, neg neg)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)#limit gradients size before training step
            optimizer.step()

            # Logging accumulators
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += v_loss.item()
            total_entropy += ent.item()
            total_approx_kl += approx_kl.item()
            total_clipfrac += clipfrac.item()
            num_updates += 1

    # Return useful training metrics
    metrics = {
        "loss": total_loss / max(num_updates, 1),
        "policy_loss": total_policy_loss / max(num_updates, 1),
        "value_loss": total_value_loss / max(num_updates, 1),
        "entropy": total_entropy / max(num_updates, 1),
        "approx_kl": total_approx_kl / max(num_updates, 1),
        "clipfrac": total_clipfrac / max(num_updates, 1),
        "batch_size": B,
        "updates": num_updates,
    }
    return metrics

def extract_episode_stats(infos, ep_returns, ep_lengths):
    if not isinstance(infos, dict):
        return

    # Pattern A: direct episode dict (sometimes present)
    if "episode" in infos and infos["episode"] is not None:
        ep = infos["episode"]
        r = ep.get("r", None)
        l = ep.get("l", None)
        if r is not None:
            for x in (r if hasattr(r, "__len__") else [r]):
                ep_returns.append(float(x))
        if l is not None:
            for x in (l if hasattr(l, "__len__") else [l]):
                ep_lengths.append(int(x))

    # Pattern B: final_info list (very common in vector envs)
    if "final_info" in infos and infos["final_info"] is not None:
        for finfo in infos["final_info"]:
            if finfo is None:
                continue
            if "episode" in finfo:
                ep = finfo["episode"]
                ep_returns.append(float(ep["r"]))
                ep_lengths.append(int(ep["l"]))

def mean_or_nan(dq):
    return float(np.mean(dq)) if len(dq) else float("nan")

@torch.no_grad()
def evaluate_agent(model, device, episodes=50, seed=123, render=False):
    model_was_training = model.training
    model.eval()

    render_mode = "human" if render else None
    print(f"Creating env with render_mode={render_mode}...")

    env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode)

    if HAS_FRAMESTACK_OBS:
        env = FrameStackObservation(env, frame_stack)
    else:
        env = FrameStack(env, num_stack=frame_stack)

    env = gym.wrappers.RecordEpisodeStatistics(env)

    episode_returns = []
    episode_lengths = []

    for ep in range(episodes):
        print(f"Starting eval episode {ep+1}/{episodes}...")
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        ep_len = 0

        while not done:
            obs_t = preprocess_obs(obs, device)   # (1, 12, 96, 96)

            mean, _ = model(obs_t)

            action = mean.clone()
            action[:, 0] = torch.tanh(action[:, 0])      # steering in [-1,1]
            action[:, 1] = torch.sigmoid(action[:, 1])   # gas in [0,1]
            action[:, 2] = torch.sigmoid(action[:, 2])   # brake in [0,1]

            action_np = action.squeeze(0).cpu().numpy()  # <- IMPORTANT

            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            ep_ret += float(reward)
            ep_len += 1

        print(f"Finished eval episode {ep+1}/{episodes}: return={ep_ret:.1f}, len={ep_len}")
        episode_returns.append(ep_ret)
        episode_lengths.append(ep_len)

    env.close()

    if model_was_training:
        model.train()

    returns = np.array(episode_returns, dtype=np.float32)
    lengths = np.array(episode_lengths, dtype=np.int32)

    results = {
        "returns": returns,
        "lengths": lengths,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "median_return": float(np.median(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "episodes": int(episodes),
    }
    return results

def plot_training_curves(history, save_dir="plots", show=False):
    os.makedirs(save_dir, exist_ok=True)
    x = history["update"]

    plt.figure()
    plt.plot(x, history["ep_ret_50"])
    plt.title("Episode Return (rolling 50)")
    plt.xlabel("Update")
    plt.ylabel("Return")
    plt.savefig(os.path.join(save_dir, "ep_return.png"), bbox_inches="tight")
    if show:
        plt.show(block=False)
        plt.pause(2)
    plt.close()

    plt.figure()
    plt.plot(x, history["entropy"])
    plt.title("Policy Entropy")
    plt.xlabel("Update")
    plt.ylabel("Entropy")
    plt.savefig(os.path.join(save_dir, "entropy.png"), bbox_inches="tight")
    if show:
        plt.show(block=False)
        plt.pause(2)
    plt.close()

    plt.figure()
    plt.plot(x, history["approx_kl"], label="approx_kl")
    plt.plot(x, history["clipfrac"], label="clipfrac")
    plt.title("PPO Diagnostics")
    plt.xlabel("Update")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "ppo_diag.png"), bbox_inches="tight")
    if show:
        plt.show(block=False)
        plt.pause(2)
    plt.close()

    print(f"✅ Saved plots to folder: {save_dir}")

def save_checkpoint(model, path=SAVE_PATH):
    torch.save(model.state_dict(), path)
    print(f"✅ Saved model weights to: {path}")

def load_checkpoint(model, device, path=SAVE_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at: {path}")
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"✅ Loaded model weights from: {path}")

def save_history(history, path=HISTORY_PATH):
    np.savez(path, **history)
    print(f"✅ Saved training history to: {path}")

def print_eval_summary(results, solved_threshold=900.0):
    mean_ret = results["mean_return"]
    std_ret = results["std_return"]
    median_ret = results["median_return"]
    min_ret = results["min_return"]
    max_ret = results["max_return"]
    mean_len = results["mean_length"]
    std_len = results["std_length"]
    episodes = results["episodes"]

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Episodes           : {episodes}")
    print(f"Mean return        : {mean_ret:.2f}")
    print(f"Std return         : {std_ret:.2f}")
    print(f"Median return      : {median_ret:.2f}")
    print(f"Min return         : {min_ret:.2f}")
    print(f"Max return         : {max_ret:.2f}")
    print(f"Mean episode length: {mean_len:.2f}")
    print(f"Std episode length : {std_len:.2f}")
    print(f"Solved threshold   : {solved_threshold:.1f}")
    print(f"Gap to solved      : {solved_threshold - mean_ret:.2f}")
    print("=" * 60)

    if mean_ret >= solved_threshold:
        print("Status: SOLVED / very strong performance")
    elif mean_ret >= 800:
        print("Status: Strong performance")
    elif mean_ret >= 600:
        print("Status: Good performance")
    elif mean_ret >= 300:
        print("Status: Moderate performance")
    else:
        print("Status: Weak / unstable performance")

    print("=" * 60)

def plot_eval_results(results, save_path="ppo_eval_50ep.png", rolling=5):
    returns = np.asarray(results["returns"])
    lengths = np.asarray(results["lengths"])
    episodes = np.arange(1, len(returns) + 1)

    mean_ret = results["mean_return"]
    std_ret = results["std_return"]
    min_ret = results["min_return"]
    max_ret = results["max_return"]
    median_ret = results["median_return"]

    # rolling mean
    if len(returns) >= rolling:
        kernel = np.ones(rolling) / rolling
        rolling_mean = np.convolve(returns, kernel, mode="valid")
        rolling_x = np.arange(rolling, len(returns) + 1)
    else:
        rolling_mean = returns
        rolling_x = episodes

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # ---- Top: returns ----
    axes[0].plot(episodes, returns, marker="o", linewidth=1, label="Episode return")
    axes[0].plot(rolling_x, rolling_mean, linewidth=2, label=f"Rolling mean ({rolling})")
    axes[0].axhline(mean_ret, linestyle="--", label=f"Mean = {mean_ret:.1f}")
    axes[0].axhline(900.0, linestyle=":", label="Solved threshold = 900")

    axes[0].set_title(
        f"Evaluation over {len(returns)} episodes\n"
        f"mean={mean_ret:.1f}, std={std_ret:.1f}, median={median_ret:.1f}, "
        f"min={min_ret:.1f}, max={max_ret:.1f}"
    )
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ---- Bottom: episode lengths ----
    axes[1].plot(episodes, lengths, marker="o", linewidth=1, label="Episode length")
    axes[1].axhline(results["mean_length"], linestyle="--", label=f"Mean length = {results['mean_length']:.1f}")
    axes[1].set_title("Episode lengths")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved evaluation plot to: {save_path}")


    np.savez(
        path,
        returns=results["returns"],
        lengths=results["lengths"],
        mean_return=results["mean_return"],
        std_return=results["std_return"],
        median_return=results["median_return"],
        min_return=results["min_return"],
        max_return=results["max_return"],
        mean_length=results["mean_length"],
        std_length=results["std_length"],
        episodes=results["episodes"],
    )
    print(f"✅ Saved evaluation results to: {path}")

def save_eval_results(results, path="ppo_eval_50ep.npz"): 
    np.savez(
        path,
        returns=results["returns"],
        lengths=results["lengths"],
        mean_return=results["mean_return"],
        std_return=results["std_return"],
        median_return=results["median_return"],
        min_return=results["min_return"],
        max_return=results["max_return"],
        mean_length=results["mean_length"],
        std_length=results["std_length"],
        episodes=results["episodes"],
    )
    print(f"✅ Saved evaluation results to: {path}")

@torch.no_grad()
def watch_agent(model, device, episodes=1, seed=9999):
    print("Starting watch mode...")
    results = evaluate_agent(
        model,
        device,
        episodes=episodes,
        seed=seed,
        render=True,
    )
    print("Finished watch mode.")
    print(f"[WATCH] mean_return={results['mean_return']:.1f} mean_len={results['mean_length']:.1f}")

def run_full_evaluation_report(model, device, episodes=50, seed=9999):
    results = evaluate_agent(
        model,
        device,
        episodes=episodes,
        seed=seed,
        render=False,
    )

    print_eval_summary(results)
    save_eval_results(results, path="ppo_eval_50ep.npz")
    plot_eval_results(results, save_path="ppo_eval_50ep.png", rolling=5)

    return results

def main(mode="train"):
    """
    mode:
      - "train": train from scratch, save model/history, run 50-episode evaluation, save plots, then watch
      - "watch": load saved model and watch a few episodes
      - "test" : load saved model and run full 50-episode evaluation only
    """
    device = pick_device()

    # Build one temp env to get action count

    temp_env = gym.make("CarRacing-v3", continuous=True)
    action_dim = temp_env.action_space.shape[0]
    temp_env.close()

    model = ActorCriticCNN(obs_channels=frame_stack * 3, action_dim=action_dim).to(device)# frames t-3,t-2,t-1,t so channel dim is 4*3

    if mode == "watch":
        load_checkpoint(model, device, SAVE_PATH)
        watch_agent(model, device, episodes=3, seed=9999)
        return

    if mode == "test":
        load_checkpoint(model, device, SAVE_PATH)
        run_full_evaluation_report(model, device, episodes=50, seed=9999)
        return

    if mode != "train":
        raise ValueError("mode must be 'train', 'watch', or 'test'")

    # -------------------------
    # TRAIN MODE
    # -------------------------
    envs = make_vec_env(num_envs=num_envs, seed=42, stack_size=frame_stack)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    next_obs, _ = envs.reset(seed=42)

    ep_returns = deque(maxlen=50)
    ep_lengths = deque(maxlen=50)

    history = {
        "update": [],
        "ep_ret_50": [],
        "ep_len_50": [],
        "entropy": [],
        "approx_kl": [],
        "clipfrac": [],
        "value_loss": [],
    }

    start_time = time.time()
    total_steps = 0

    pbar = tqdm(range(num_updates), desc="PPO", ncols=120)
    for update in pbar:
        batch, next_obs, last_done, last_value, infos = collect_rollout(
            envs, model, device, horizon=horizon, next_obs=next_obs
        )

        extract_episode_stats(infos, ep_returns, ep_lengths)

        adv, ret = compute_gae(
            batch["rewards"],
            batch["values"],
            batch["dones"],
            last_value,
            gamma=gamma,
            lam=gae_lambda,
        )
        batch["advantages"] = adv
        batch["returns"] = ret

        metrics = train_step(
            model,
            optimizer,
            batch,
            ppo_epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            clip_coef=clip_coef,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
        )

        total_steps += envs.num_envs * horizon
        fps = int(total_steps / (time.time() - start_time + 1e-8))

        ep_ret_50 = mean_or_nan(ep_returns)
        ep_len_50 = mean_or_nan(ep_lengths)

        history["update"].append(update)
        history["ep_ret_50"].append(ep_ret_50)
        history["ep_len_50"].append(ep_len_50)
        history["entropy"].append(metrics["entropy"])
        history["approx_kl"].append(metrics["approx_kl"])
        history["clipfrac"].append(metrics["clipfrac"])
        history["value_loss"].append(metrics["value_loss"])

        pbar.set_postfix({
            "epRet50": f"{ep_ret_50:7.1f}" if not np.isnan(ep_ret_50) else "  nan  ",
            "ent": f"{metrics['entropy']:.2f}",
            "kl": f"{metrics['approx_kl']:.3f}",
            "clip": f"{metrics['clipfrac']:.2f}",
            "v": f"{metrics['value_loss']:.2f}",
            "fps": fps,
        })

        if (update + 1) % 100 == 0:
            save_checkpoint(model, SAVE_PATH)

    envs.close()

    save_checkpoint(model, SAVE_PATH)
    save_history(history, HISTORY_PATH)

    run_full_evaluation_report(
        model,
        device,
        episodes=50,
        seed=9999,
    )

    plot_training_curves(history, save_dir="plots", show=False)

    watch_agent(model, device, episodes=3, seed=9999)
  
if __name__ == "__main__":
    main(mode="watch")