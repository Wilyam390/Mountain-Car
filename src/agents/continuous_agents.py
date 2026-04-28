import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from collections import deque

# Note: Ensure .base_agent is available in your local directory
from .base_agent import BaseAgent

# ============================================================================
# NEURAL NETWORK ARCHITECTURES
# ============================================================================

class QNetwork(nn.Module):
    """Neural network for DQN (discretized actions)."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        return self.net(state)


class StochasticPolicy(nn.Module):
    """Stochastic policy for REINFORCE and A2C (outputs distribution)."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.common = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        # log_std as a trainable parameter for stability
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.common(state)
        # Tanh ensures the mean is within [-1, 1]
        mean = torch.tanh(self.mean_layer(x))
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mean, std


class DeterministicPolicy(nn.Module):
    """Deterministic policy for DDPG (Actor)."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh() # Single Tanh here to bound action to [-1, 1]
        )

    def forward(self, state):
        return self.net(state)


class ValueNetwork(nn.Module):
    """Critic/Baseline network for A2C and REINFORCE."""
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


# ============================================================================
# DQN AGENT
# ============================================================================
class DQNAgent(BaseAgent):
    """DQN Agent: Discretizes the continuous action space."""
    def __init__(
        self,
        n_actions: int = 11,
        state_shape: Tuple = (2,),
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        update_freq: int = 4,
        device: str = "cpu",
    ):
        super().__init__(n_actions=n_actions, state_shape=state_shape, agent_name="DQNAgent")
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma, self.batch_size = gamma, batch_size
        self.update_freq = update_freq
        self.device = torch.device(device)
        self.epsilon = epsilon_start
        self.epsilon_min, self.epsilon_decay = epsilon_min, epsilon_decay

        self.q_net = QNetwork(state_shape[0], n_actions, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_shape[0], n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)

        self.buffer = deque(maxlen=buffer_size)
        self.action_bins = np.linspace(-1, 1, n_actions)
        self.train_step_count = 0

    def get_hyperparams(self) -> Dict[str, Any]:
        return {"n_actions": self.n_actions, "epsilon": self.epsilon, "gamma": self.gamma}

    def set_hyperparams(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key): setattr(self, key, value)

    def save(self, path: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), path / "q_net.pth")

    def load(self, path: str) -> None:
        path = Path(path)
        self.q_net.load_state_dict(torch.load(path / "q_net.pth", map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())

    def train_step(self, state: np.ndarray) -> Tuple[np.ndarray, Dict]:
        if np.random.rand() < self.epsilon:
            idx = np.random.randint(self.n_actions)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                idx = self.q_net(state_t).argmax(1).item()
        return np.array([self.action_bins[idx]], dtype=np.float32), {"epsilon": self.epsilon}

    def learn(self, state, action, reward, next_state, done):
        idx = np.argmin(np.abs(self.action_bins - action[0]))
        self.buffer.append((state, idx, reward, next_state, done))
        self.train_step_count += 1
        if len(self.buffer) >= self.batch_size and self.train_step_count % self.update_freq == 0:
            self._update()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _update(self):
        batch = [self.buffer[i] for i in np.random.choice(len(self.buffer), self.batch_size)]
        s, a, r, ns, d = zip(*batch)
        s, a = torch.FloatTensor(np.array(s)).to(self.device), torch.LongTensor(a).to(self.device).unsqueeze(1)
        r, ns, d = torch.FloatTensor(r).to(self.device), torch.FloatTensor(np.array(ns)).to(self.device), torch.FloatTensor(d).to(self.device)
        with torch.no_grad():
            next_q = self.target_net(ns).max(1)[0]
            target = r + (1 - d) * self.gamma * next_q
        curr_q = self.q_net(s).gather(1, a).squeeze(1)
        loss = nn.MSELoss()(curr_q, target)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        if self.train_step_count % 1000 == 0: self.target_net.load_state_dict(self.q_net.state_dict())

    def act(self, state, training=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad(): idx = self.q_net(state_t).argmax(1).item()
        return np.array([self.action_bins[idx]], dtype=np.float32)
    
# ============================================================================
# REINFORCE AGENT
# ============================================================================
class REINFORCEAgent(BaseAgent):
    def __init__(self, state_shape=(2,), action_shape=(1,), hidden_dim=128, learning_rate=0.0005, gamma=0.99, entropy_coef=0.01, device="cpu"):
        super().__init__(n_actions=1, state_shape=state_shape, agent_name="REINFORCEAgent")
        self.device = torch.device(device)
        self.gamma, self.entropy_coef = gamma, entropy_coef
        self.learning_rate = learning_rate
        
        self.policy = StochasticPolicy(state_shape[0], action_shape[0], hidden_dim).to(self.device)
        self.baseline = ValueNetwork(state_shape[0], hidden_dim).to(self.device)
        self.optimizer_p = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.optimizer_b = optim.Adam(self.baseline.parameters(), lr=learning_rate * 2)
        self.ep_states, self.ep_actions, self.ep_rewards = [], [], []

    def get_hyperparams(self) -> Dict[str, Any]:
        return {"learning_rate": self.learning_rate, "gamma": self.gamma, "entropy_coef": self.entropy_coef}

    def set_hyperparams(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key): setattr(self, key, value)

    def save(self, path: str) -> None:
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), path / "policy.pth")
        torch.save(self.baseline.state_dict(), path / "baseline.pth")

    def load(self, path: str) -> None:
        path = Path(path)
        self.policy.load_state_dict(torch.load(path / "policy.pth", map_location=self.device))
        self.baseline.load_state_dict(torch.load(path / "baseline.pth", map_location=self.device))

    def train_step(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, std = self.policy(state_t)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action_clamped = torch.clamp(action, -1, 1)
        self.ep_states.append(state)
        self.ep_actions.append(action_clamped.detach().cpu().numpy().flatten())
        return action_clamped.detach().cpu().numpy().flatten(), {"std": std.mean().item()}

    def learn(self, state, action, reward, next_state, done):
        self.ep_rewards.append(reward)
        if done: self._update(); self.ep_states, self.ep_actions, self.ep_rewards = [], [], []

    def _update(self):
        R = 0; returns = []
        for r in reversed(self.ep_rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        states = torch.FloatTensor(np.array(self.ep_states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.ep_actions)).to(self.device)
        vals = self.baseline(states)
        advantages = returns - vals.detach()
        mean, std = self.policy(states)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        policy_loss = -(log_probs * advantages).mean()
        entropy_loss = -self.entropy_coef * dist.entropy().mean()
        self.optimizer_p.zero_grad(); (policy_loss + entropy_loss).backward(); self.optimizer_p.step()
        baseline_loss = nn.MSELoss()(vals, returns)
        self.optimizer_b.zero_grad(); baseline_loss.backward(); self.optimizer_b.step()

    def act(self, state, training=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad(): mean, _ = self.policy(state_t)
        return mean.cpu().numpy().flatten()
    
# ============================================================================
# A2C AGENT
# ============================================================================
class A2CAgent(BaseAgent):
    def __init__(self, state_shape=(2,), action_shape=(1,), hidden_dim=128, learning_rate=0.0007, gamma=0.99, entropy_coef=0.01, device="cpu"):
        super().__init__(n_actions=1, state_shape=state_shape, agent_name="A2CAgent")
        self.device = torch.device(device)
        self.gamma, self.entropy_coef = gamma, entropy_coef
        self.learning_rate = learning_rate

        self.policy = StochasticPolicy(state_shape[0], action_shape[0], hidden_dim).to(self.device)
        self.value = ValueNetwork(state_shape[0], hidden_dim).to(self.device)
        self.optimizer_p = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.optimizer_v = optim.Adam(self.value.parameters(), lr=learning_rate * 2)

    def get_hyperparams(self) -> Dict[str, Any]:
        return {"learning_rate": self.learning_rate, "gamma": self.gamma, "entropy_coef": self.entropy_coef}

    def set_hyperparams(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key): setattr(self, key, value)

    def save(self, path: str) -> None:
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), path / "policy.pth")
        torch.save(self.value.state_dict(), path / "value.pth")

    def load(self, path: str) -> None:
        path = Path(path)
        self.policy.load_state_dict(torch.load(path / "policy.pth", map_location=self.device))
        self.value.load_state_dict(torch.load(path / "value.pth", map_location=self.device))

    def train_step(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, std = self.policy(state_t)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
        return torch.clamp(action, -1, 1).cpu().numpy().flatten(), {"std": std.mean().item()}

    def learn(self, state, action, reward, next_state, done):
        s_t, ns_t = torch.FloatTensor(state).unsqueeze(0).to(self.device), torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        a_t = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        val = self.value(s_t)
        with torch.no_grad():
            next_val = self.value(ns_t) if not done else 0
            td_target = reward + self.gamma * next_val
            advantage = td_target - val.item()
        mean, std = self.policy(s_t)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(a_t).sum()
        policy_loss = -(log_prob * advantage)
        entropy_loss = -self.entropy_coef * dist.entropy().sum()
        self.optimizer_p.zero_grad(); (policy_loss + entropy_loss).backward(); self.optimizer_p.step()
        val_pred = self.value(s_t)
        value_loss = nn.MSELoss()(val_pred, torch.tensor([td_target]).to(self.device))
        self.optimizer_v.zero_grad(); value_loss.backward(); self.optimizer_v.step()

    def act(self, state, training=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad(): mean, _ = self.policy(state_t)
        return mean.cpu().numpy().flatten()
    
# ============================================================================
# DDPG AGENT
# ============================================================================

class DDPGAgent(BaseAgent):
    """Deep Deterministic Policy Gradient (DDPG) - Fully Abstract-Compliant."""
    def __init__(
        self,
        state_shape: Tuple = (2,),
        action_shape: Tuple = (1,),
        hidden_dim: int = 256,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        tau: float = 0.005,
        noise_scale: float = 0.2,
        buffer_size: int = 50000,
        batch_size: int = 128,
        device: str = "cpu"
    ):
        # Ensure we pass the expected arguments to the BaseAgent constructor
        super().__init__(n_actions=1, state_shape=state_shape, agent_name="DDPGAgent")
        
        self.device = torch.device(device)
        self.action_shape = action_shape
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_scale = noise_scale

        # Networks
        self.actor = DeterministicPolicy(state_shape[0], action_shape[0], hidden_dim).to(self.device)
        self.actor_target = DeterministicPolicy(state_shape[0], action_shape[0], hidden_dim).to(self.device)
        
        self.critic = nn.Sequential(
            nn.Linear(state_shape[0] + action_shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        self.critic_target = nn.Sequential(
            nn.Linear(state_shape[0] + action_shape[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=learning_rate * 0.1)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.buffer = deque(maxlen=buffer_size)

    # --- Abstract Method Implementations ---

    def get_hyperparams(self) -> Dict[str, Any]:
        """REQUIRED by BaseAgent: Return dict of current params."""
        return {
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.critic_opt.param_groups[0]['lr'],
            "gamma": self.gamma,
            "tau": self.tau,
            "noise_scale": self.noise_scale,
            "batch_size": self.batch_size
        }

    def set_hyperparams(self, params: Dict[str, Any]) -> None:
        """REQUIRED by BaseAgent: Update params from dict."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save(self, path: str) -> None:
        """REQUIRED by BaseAgent: Save weights to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path / "actor.pth")
        torch.save(self.critic.state_dict(), path / "critic.pth")

    def load(self, path: str) -> None:
        """REQUIRED by BaseAgent: Load weights from disk."""
        path = Path(path)
        self.actor.load_state_dict(torch.load(path / "actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(path / "critic.pth", map_location=self.device))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    # --- Core Logic ---

    def train_step(self, state: np.ndarray) -> Tuple[np.ndarray, Dict]:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().flatten()
        
        noise = np.random.normal(0, self.noise_scale, size=action.shape)
        action_out = np.clip(action + noise, -1, 1).astype(np.float32)
        return action_out, {"noise": np.mean(np.abs(noise))}

    def learn(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) >= self.batch_size:
            self._update()

    def _update(self):
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        s, a, r, ns, d = zip(*batch)
        
        s = torch.FloatTensor(np.array(s)).to(self.device)
        a = torch.FloatTensor(np.array(a)).to(self.device)
        r = torch.FloatTensor(r).to(self.device).unsqueeze(1)
        ns = torch.FloatTensor(np.array(ns)).to(self.device)
        d = torch.FloatTensor(d).to(self.device).unsqueeze(1)

        # Critic update
        with torch.no_grad():
            na = self.actor_target(ns)
            target_q = r + (1 - d) * self.gamma * self.critic_target(torch.cat([ns, na], dim=1))
        
        curr_q = self.critic(torch.cat([s, a], dim=1))
        critic_loss = nn.MSELoss()(curr_q, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        actor_loss = -self.critic(torch.cat([s, self.actor(s)], dim=1)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft updates
        with torch.no_grad():
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def act(self, state: np.ndarray, training: bool = False) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.actor(state_t).cpu().numpy().flatten()