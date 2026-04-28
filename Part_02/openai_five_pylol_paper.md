# Large-Scale Deep Reinforcement Learning for MOBA Games: A Dissection of OpenAI Five and PyLoL

This paper provides a technical dissection of how OpenAI built and trained OpenAI Five, the first AI system to defeat esports world champions, and contrasts it with PyLoL, an open-source attempt to replicate the same approach for League of Legends. We analyze the architectural choices, distributed training infrastructure, reward engineering, self-play dynamics, and the fundamental scaling hypothesis that underpinned both systems. We also examine where PyLoL diverges from OpenAI's approach due to practical constraints, and what this reveals about the gap between frontier ML research and accessible reproducibility.
---

## Abstract

This paper provides a technical dissection of how OpenAI built and trained OpenAI Five, the first AI system to defeat esports world champions, and contrasts it with PyLoL, an open-source attempt to replicate the same approach for League of Legends. We analyze the architectural choices, distributed training infrastructure, reward engineering, self-play dynamics, and the fundamental scaling hypothesis that underpinned both systems. We also examine where PyLoL diverges from OpenAI's approach due to practical constraints, and what this reveals about the gap between frontier ML research and accessible reproducibility.

---

## 1. Introduction: Why MOBAs Are Hard

Multiplayer Online Battle Arena (MOBA) games represent one of the most difficult categories of environments ever posed to reinforcement learning. Unlike Chess, Go, or even Atari games, MOBAs exhibit a pathological combination of properties that defeat most standard RL approaches:

- **Long temporal horizons**: A Dota 2 game runs at 30 fps for approximately 45 minutes. At one decision every 4 frames, that is ~20,000 steps per episode. Chess averages 80 moves; Go averages 150.
- **Partial observability**: Each team observes only the portion of the map near their own units. The rest is "fog of war" — the agent must model an adversary it cannot see.
- **High-dimensional discrete-continuous action spaces**: On a given timestep, OpenAI Five selects from between 8,000 and 80,000 possible actions depending on hero, encoding movement targets, ability targets, item use, and more.
- **High-dimensional observation spaces**: ~16,000 float and categorical values per timestep encoding hero positions, health, mana, items, spell cooldowns, game economy, and more.
- **Multi-agent coordination**: Five heroes must act simultaneously and cooperatively, each making independent decisions with nearly identical observations but needing to avoid duplicating or conflicting actions.
- **Non-stationarity**: The opponent adapts. Self-play means the distribution of "enemies" changes as the agent improves, making the learning target perpetually moving.

The lesson from OpenAI Five is that none of these challenges required a fundamentally new algorithm to overcome. The key insight was that scale was the solution.

---

## 2. OpenAI Five: Technical Architecture

### 2.1 The Scaling Hypothesis

The central thesis of the OpenAI Five project was explicit and deliberate: **existing RL algorithms, scaled sufficiently, are sufficient to achieve superhuman performance on complex sequential decision-making tasks**. The team used Proximal Policy Optimization (PPO) trained over 10 months on thousands of GPUs.

The system consumed approximately **770 ± 50 PFLOPs/s·days** of total compute. For context, this is:
- ~50–150× larger batch sizes than AlphaGo
- ~20× larger model
- ~25× longer training time


### 2.2 The Policy Network

The policy is a function π: history of observations → probability distribution over actions.

It is parameterized as a **recurrent neural network with ~159 million parameters**, dominated by a **single-layer 4096-unit LSTM** (Long Short-Term Memory). The LSTM constitutes 84% of total parameter count. The architecture is deliberately conservative:

```
Observation (16,000 values)
    ↓
Observation Processing (entity embeddings, pooling, concatenation)
    ↓
4096-unit LSTM (hidden state maintained across timesteps)
    ↓
Policy Head (softmax over action space)
Value Head (scalar baseline estimate)
```

Key architectural choices to note:

**Semantic observation space:** The system receives structured game state data directly from the engine API rather than rendering frames to pixels. This was pragmatically necessary — rendering every training game would have multiplied compute requirements many-fold — but also philosophically consistent with studying strategic decision-making over vision.

**Shared parameters across heroes:** The same policy network controls all five heroes on a team, with separate hidden LSTM states per hero. The hero identity is passed as an extra input to allow the network to differentiate, but the weights are shared. This is a massive parameter efficiency gain and implicitly enforces transferable strategy representations.

**No Monte Carlo Tree Search:** Unlike AlphaGo, OpenAI Five uses no lookahead search at inference time. Every action is sampled directly from the policy output at each timestep. This was a deliberate design choice to keep inference tractable at scale during rollouts.

### 2.3 The Observation Space

The observation vector (~16,000 values per timestep) encodes:
- All unit states (position, health, mana, status effects, attack/ability properties)
- All building states
- Game economy (gold, net worth, kill/death/assist counts)
- Ability and item cooldowns
- Team fog-of-war visibility flags
- Time and game phase features
- Derived relational features (e.g., distance between units)

Categorical values with many possibilities (hero types, item types, ability types) are encoded as integer indices and embedded into continuous vectors during observation processing.

### 2.4 The Action Space

Actions are discretized on each timestep. The agent outputs a hierarchical action:

1. **Action type:** move, attack, use ability 1–4, use item 1–6, purchase item, etc.
2. **Target:** either a unit/building (selected from all valid targets) or a map location (discretized grid)

The combinatorial product of these choices yields 8,000–80,000 valid actions depending on hero state. Certain decisions were hand-scripted and excluded from the learned policy:
- Item purchase order
- Courier control (a shared unit)
- Item stash management

This pragmatic simplification was explicitly acknowledged but superhuman performance was achieved without them.

### 2.5 The Reward Function

The reward signal was shaped (not purely sparse wins/losses). The reward function was designed once at project start by team members with domain expertise and only minimally tweaked thereafter.

Reward components include:
- **Win/loss**: +1 / −1 at game termination
- **Kill/death**: reward for killing enemy heroes, penalty for dying
- **Gold income**: reward for farm efficiency
- **Tower and building damage**: reward for objective progress

Crucially, the reward is **symmetrized**: the agent's reward is computed as its own shaped reward *minus* the opposing team's shaped reward. This zero-sum symmetrization ensures that the agent cannot increase reward simply by mutual farming with the opponent.

The team found that additional shaped signals (beyond pure win/loss) were critical for successful training, particularly early in training when sparse win/loss would provide almost no learning signal in very long episodes.

### 2.6 The Training Algorithm: PPO at Scale

The optimization backbone is **Proximal Policy Optimization (PPO)** with **Generalized Advantage Estimation (GAE)**:

- **GAE** reduces variance in advantage estimates by exponentially blending multi-step returns with λ = 0.95.
- **PPO** clips the policy update ratio to within [1−ε, 1+ε] to prevent destructively large policy updates.
- **Truncated BPTT**: Backpropagation through the LSTM is truncated to 16-timestep windows to make gradient computation tractable.
- **Adam optimizer** with per-parameter gradient clipping.

The effective batch size was enormous: 120 samples × 16 timesteps × up to 1,536 optimizer GPUs = **~2.95 million timesteps per gradient update**, refreshed every ~2 seconds.

Gradients were averaged across optimizer GPUs using **NCCL2 allreduce** before synchronous application. Parameters were published to a central **Redis** store called the "controller" every 32 gradient steps.

---

## 3. OpenAI Five: Distributed System Architecture

The training system ("Rapid") is a custom distributed platform running on **Google Cloud Platform**. It consists of four machine types operating in a tight asynchronous loop:

```
┌──────────────────────────────────────────────────────────┐
│                   Controller (Redis)                     │
│          Parameter storage + system metadata             │
└──────┬───────────────────────────────────────┬───────────┘
       │ publish params (every 32 grad steps)  │ pull params
       ↓                                       ↓
┌─────────────┐   observations     ┌──────────────────────┐
│   Rollout   │ ◄────────────────► │  Forward Pass GPUs   │
│  Workers    │   sampled actions  │  (batch ~60 obs)     │
│  (CPU game  │                    └──────────────────────┘
│  instances) │                              │
└──────┬──────┘                              │
       │ game data (async, every 256 steps)  │
       ↓                                     │
┌─────────────────────────────────────────────────────────┐
│                   Optimizer GPUs                        │
│  (up to 1536 GPUs, NCCL2 allreduce, experience buffers) │
└─────────────────────────────────────────────────────────┘
```

### Rollout Workers (CPU)
- Run the Dota 2 game engine directly (Linux-native binary, no rendering)
- Execute games at ~0.5× real-time speed to maximize throughput
- Do not run the policy themselves — they delegate inference to Forward Pass GPUs
- 80% of games play current policy vs. itself; 20% play against older policies (opponent pool)
- Push data asynchronously every 256 timesteps (~34 seconds of gameplay)

### Forward Pass GPUs
- Receive batches of ~60 observations from rollout workers
- Run policy forward pass, return sampled actions
- Frequently poll controller for latest parameters

### Optimizer GPUs
- Maintain local experience buffers fed asynchronously by rollout workers
- Sample random minibatches, compute gradients
- Allreduce gradients across all optimizer GPUs (NCCL2)
- Apply synchronized updates to parameters
- Publish new parameters to controller

This architecture achieves the critical goal of keeping the rollout-optimization loop as tight as possible, minimizing the "staleness" of experience used for updates.

### Scale During Training
- At peak: ~1,536 optimizer GPUs, thousands of rollout machines
- System consumed ~2 million frames of experience every 2 seconds
- Total training duration: 10 months (June 2018 – April 2019), ~180 effective training days

---

## 4. Self-Play and Opponent Sampling

OpenAI Five was trained almost entirely through self-play where the agent plays against copies of itself. This sidesteps the need for hand-crafted opponents or human demonstration data.

The opponent sampling strategy:
- **80% of games:** current policy vs. current policy
- **20% of games:** current policy vs. a randomly sampled older version of itself

The 20% historical opponent games serve a regularization function: they prevent the agent from becoming too narrowly specialized against its current self, which could lead to cyclic strategy dynamics (rock-paper-scissors style). The opponent pool was sampled uniformly at random from a window of recent historical policies.

**TrueSkill** was used as the automated evaluation metric throughout training, enabling continuous monitoring of skill without expensive human evaluation. A TrueSkill difference of ~8.3 corresponds to an ~80% winrate.

---

## 5. The Surgery System: Continual Training Under a Moving Target

One of the most practically significant contributions of OpenAI Five was the **surgery** framework for continual learning under non-stationary environments.

Over 10 months, three categories of change forced the training environment to shift:
1. **Research-driven changes:** New reward components, architecture changes, observation additions
2. **Scope expansion:** Adding new game mechanics (e.g., the "buyback" action) as engineering progressed
3. **Game version updates:** Valve periodically patched Dota 2, changing hero/item properties

Without surgery, each such change would require retraining from scratch which is a 2-month process per attempt. Over ~20 major changes in 10 months, this would have been impossible.

Surgery is a collection of offline parameter transformation tools satisfying:

> π̂_θ̂(o) = π_θ(o) for all observations o

That is, the new model in the new environment implements the same function as the old model in the old environment, preserving skill across the transformation.

Concrete surgery operations include:
- **Adding observations:** Zero-initialize new embedding weights; no change to existing logic
- **Expanding layers:** Net2Net-style width expansion — new neurons initialized as linear combinations of existing ones
- **Action space changes:** Map old action indices to new; initialize novel action outputs to small random values
- **Architecture changes:** Distillation or approximate function matching when exact preservation is impossible

The validation experiment ("Rerun") confirmed that training from scratch with the final configuration achieved higher ultimate performance than surgery-based OpenAI Five, but required 2 months even in the clean setting, confirming surgery's critical role in the iterative development process. Rerun ultimately exceeded OpenAI Five's skill level, achieving >98% winrate against the final OpenAI Five agent.

---

## 6. PyLoL: The Open-Source Echo

### 6.1 What PyLoL Is

PyLoL (by MiscellaneousStuff on GitHub) is the Python component of the **League of Legends v4.20 Reinforcement Learning Environment (LoLRLE)**. It explicitly mirrors the design philosophy of OpenAI Five applied to League of Legends, and its architecture is directly inspired by **PySC2**, DeepMind's StarCraft II Python environment, itself inspired by the same paradigm.

The stack consists of:
- **LeagueSandbox**: An open-source reimplementation of the League of Legends v4.20 game server (C#/.NET Core), which exposes a bot-scriptable API
- **PyLoL**: A Python RL environment wrapper exposing an OpenAI Gym-compatible interface
- **Redis**: Used as the communication bridge between the game server and the Python agent (mirroring OpenAI Five's use of Redis as its controller store)

### 6.2 How PyLoL Mirrors OpenAI Five's Approach

| Component | OpenAI Five | PyLoL |
|---|---|---|
| Environment interface | Custom Dota 2 API (Valve Bot Scripting) | LeagueSandbox game server API |
| RL interface | Custom PPO + Rapid | OpenAI Gym compatible |
| Policy algorithm | PPO + LSTM | PPO + LSTM (via lolgym fork) |
| Communication | Redis (controller) | Redis (game server ↔ Python) |
| Observation type | Semantic game state API | Semantic game state API |
| Self-play | Yes, central to training | Limited (1v1 adversarial) |
| Scale | Thousands of GPUs | Single machine / Google Colab |

The **lolgym** fork (Eyceoz & Lee, Columbia University) built directly on top of PyLoL to implement the full PPO training loop, using an adversarial multi-agent setup where two agents play against each other in 1v1, a micro-scale echo of OpenAI Five's self-play.

### 6.3 Observation and Action Space

PyLoL's observations encode the information available to a human player:
- Champion position, health, mana, level
- Nearby unit states (minions, jungle camps, enemy champions)
- Ability and item cooldowns
- Map/game state features (tower health, gold, time)

The action space covers:
- Movement (discretized map coordinates)
- Basic attack (target selection)
- Ability casts (Q, W, E, R with optional targeting)
- Item use
- No-ops

In a 1v1 constrained setting, this is significantly smaller than OpenAI Five's full 5v5 space, but the structural challenge (long horizons, partial information, continuous positioning) remains intact.

### 6.4 The Critical Constraint: No Official API

The most fundamental divergence from OpenAI Five is environmental. Valve provides an official **Bot Scripting API** for Dota 2, allowing external programs to receive game state and issue actions programmatically. Riot Games provides **no such API** for League of Legends.

This forced PyLoL to use a reverse-engineered, open-source reimplementation of the v4.20 game server called **LeagueSandbox** that:
- Is several years behind the current live game
- Lacks many game mechanics (minions in some configurations, jungle complexity, etc.)
- Has undefined behavior in edge cases due to incomplete reimplementation
- Cannot be scaled to match Riot's live servers

This single constraint is why PyLoL cannot achieve what OpenAI Five achieved at scale.

### 6.5 Training Results

Using PyLoL + lolgym with PPO, the first published results showed:
- The agent learned to achieve 'first blood' on a stationary/passive opponent
- It demonstrated basic kiting behavior (maintaining distance while attacking)
- It outperformed random agents and hand-scripted baseline bots in 1v1

This is qualitatively comparable to where OpenAI Five was in the very earliest stages of its training which is learning basic combat before any strategic understanding emerged.

The scale gap is vast: OpenAI Five trained for 180 days at PFlop/s-day scale. PyLoL agents train on a single machine in Google Colab, representing approximately 6–7 orders of magnitude less compute.

---

## 7. Comparative Analysis

### 7.1 Algorithmic Similarity

Both systems share the same algorithmic core: PPO + LSTM + self-play. This is not coincidental — PPO was selected by OpenAI because it was the first algorithm to show reliable early learning progress in the Dota environment, and it became the de facto standard for on-policy MOBA RL. The LSTM is necessary because both games require temporal memory: understanding what has happened recently (e.g., where an enemy was last seen) is essential for rational decision-making under partial observability.

### 7.2 The Role of Scale

The gap between PyLoL's results and OpenAI Five's is almost entirely explained by compute and data scale, not algorithmic sophistication:

| Factor | OpenAI Five | PyLoL |
|---|---|---|
| Compute | ~770 PFlop/s-days | <1 GPU-day |
| Frames/second | ~2 million/sec | ~100s/sec |
| Training duration | 180 days | Hours to days |
| Model parameters | 159M | ~1–10M |
| Self-play opponents | Large historical pool | Single opponent |

The scaling hypothesis — that raw compute applied to a correct algorithmic framework can produce superhuman behavior — is validated by OpenAI Five and implicitly tested at the opposite extreme by PyLoL.

### 7.3 Infrastructure as a First-Class Research Artifact

OpenAI Five's paper dedicates substantial space to infrastructure: Rapid, surgery, the Redis controller, the rollout/forward-pass/optimizer separation. This is not incidental — **the engineering of the distributed training system was a primary research contribution**, not a footnote.

PyLoL makes this concrete by negative example: even with the correct algorithm, without infrastructure to generate experience at scale, learning cannot progress beyond basic behaviors. The game environment bottleneck (no official API, slow LeagueSandbox) is the proximate constraint, but the deeper lesson is that **ML research at this scale is inseparable from systems engineering**.

### 7.4 Reward Engineering

Both systems require careful reward shaping. OpenAI Five's shaped reward (kill/death/gold/buildings minus opponent's version) was constructed by domain experts and kept stable throughout training. This stability was deliberate: the team recognized that reward function changes require surgery or retraining.

PyLoL's reward is necessarily simpler — typically:
- +1 for killing the enemy champion
- −1 for dying
- Small per-timestep reward for dealing damage

The absence of economy and objective rewards in PyLoL's simpler setup means the agent cannot learn late-game macro strategy — but in 1v1 without full game mechanics, this is appropriate scoping.

---

## 8. Lessons and Broader Implications

### 8.1 Scale is a First-Class Research Variable

OpenAI Five demonstrated empirically that scaling compute, data, and model size through a correct algorithmic framework can solve problems once considered beyond RL's reach. This foreshadowed the scaling laws research that would later dominate LLM development. 

### 8.2 Continual Learning is Underrated

The surgery framework addressed a problem most RL research ignores: real environments change. Code gets updated, game patches are released, reward functions evolve. The ability to maintain a high-skill agent across these changes without retraining from scratch is not just a practical convenience but it is a prerequisite for deploying RL systems in any real-world setting where the environment is not frozen.

### 8.3 The Infrastructure Bottleneck in Open Research

PyLoL illustrates that the algorithmic knowledge to train superhuman game-playing agents is now essentially public domain. PPO + LSTM + self-play is documented in papers and open-source code. What remains scarce is:
1. Environment access (official APIs or efficient simulators)
2. Compute (thousands of GPUs over months)
3. Engineering effort (distributed training infrastructure)

This creates a significant asymmetry: the intellectual knowledge is democratized, but the capability remains concentrated at institutions with resources. PyLoL is a sincere attempt to reduce this gap within the constraints of what a single developer can build.

### 8.4 The Semantic vs. Pixel Observation Question

Both systems chose semantic game-state observations over pixel-based inputs. This was pragmatic (rendering at training scale is infeasible), but it also means neither system actually learned to perceive the game as humans do. A genuine test of the full problem, which is to learn perception and strategy simultaneously from raw visual input remains largely unsolved for MOBAs, and is what the TLoL project (the successor to PyLoL) attempted to address via replay-based supervised pretraining.

---

## 9. Conclusion

OpenAI Five and PyLoL represent two points on the same trajectory: applying deep reinforcement learning to MOBA games via self-play, shaped rewards, and PPO+LSTM policies. They differ not in algorithmic approach but in scale — by approximately seven orders of magnitude in compute — and in environmental infrastructure, where Valve's open Bot API enabled OpenAI and Riot's closed ecosystem forced PyLoL into a constrained simulation.

The core lesson is not that OpenAI discovered a novel algorithm for MOBA games. They discovered that **the problem is a large-scale distributed systems engineering challenge as much as a research challenge**, and this unlocked what pure algorithmic innovation had not. PyLoL inherits this framing faithfully, and its limitations illuminate exactly how much of OpenAI Five's success was architectural and infrastructural rather than purely scientific.

---

## References

1. Berner, C. et al. (OpenAI). "Dota 2 with Large Scale Deep Reinforcement Learning." arXiv:1912.06680 (2019).
2. Schulman, J. et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347 (2017).
3. Schulman, J. et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation." arXiv:1506.02438 (2015).
4. MiscellaneousStuff. "PyLoL: League of Legends v4.20 RL Environment." GitHub (2021). https://github.com/MiscellaneousStuff/pylol
5. Eyceoz, M. & Lee, J. "lolgym: Adversarial Multi-Agent RL for League of Legends." Columbia University / GitHub (2021). https://github.com/jjlee0802cu/lolgym
6. MiscellaneousStuff. "TLoL: Human Level in League of Legends using Deep Learning." Blog post (2021). https://miscellaneousstuff.github.io
7. Vinyals, O. et al. (DeepMind). "Grandmaster level in StarCraft II using multi-agent reinforcement learning." Nature 575 (2019).
8. Chen, T. et al. "Net2Net: Accelerating Learning via Knowledge Transfer." arXiv:1511.05641 (2015).
9. Hochreiter, S. & Schmidhuber, J. "Long Short-Term Memory." Neural Computation 9(8) (1997).
10. He, K. et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
