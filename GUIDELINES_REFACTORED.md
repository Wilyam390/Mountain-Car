# Mountain Car RL Study — Research Outline

## PROBLEM OVERVIEW

**Goal**: Systematic investigation of RL methods across different state representations, algorithms, reward designs, and continuous/discrete control paradigms on the Mountain Car problem.

**Problem Characteristics**:
- Continuous state space (position, velocity)
- Discrete actions (primary) / Continuous actions (secondary)
- Non-obvious solution (momentum strategy required)
- Sparse reward structure
- Long-horizon planning needed (200 steps)

---

## PHASE STRUCTURE

### PHASE 0: CLOSED-FORM SOLUTIONS
**Investigation**: Establish theoretical baselines — deterministic policies that solve Mountain Car without learning.

**What we study**:
- Velocity envelope-based policy for discrete MountainCar-v0
- Momentum-exploitation policy for continuous MountainCarContinuous-v0
- Comparison of efficiency between discrete and continuous approaches

---

### PHASE 1: ENVIRONMENT & BASELINES
**Investigation**: Understand the problem and establish experimental infrastructure.

**What we study**:
- Environment properties and state/action space dynamics
- State discretization mechanics (baseline exploration)
- Q-learning baseline performance
- Training loop and evaluation pipeline

---

### PHASE 2: STATE DISCRETIZATION & REPRESENTATION ANALYSIS
**Investigation**: How does state representation resolution affect learning performance?

**What we study**:
- Discretization level trade-off (multiple bin counts)
- Performance vs. computational cost curves
- Convergence patterns across different resolutions
- State visitation efficiency
- Optimal discretization selection under budget constraints

---

### PHASE 3: REWARD ENGINEERING & SCENARIO DESIGN
**Investigation**: How do different reward designs affect learned behavior and convergence?

**What we study**:
- Multiple reward scenarios (e.g., min-steps vs. min-fuel)
- Behavioral differences across reward structures
- Action frequency and policy efficiency
- Learning difficulty under different incentives
- Reward shaping impact on learning dynamics

---

### PHASE 4: TABULAR METHODS COMPARISON
**Investigation**: Which classical tabular RL algorithm is best for this problem?

**What we study**:
- Off-policy vs. on-policy TD learning (different update rules)
- Monte Carlo methods (alternative value estimation)
- Hyperparameter sensitivity for each method
- Convergence speed and stability
- Robustness across random seeds

---

### PHASE 5: HYPERPARAMETER SENSITIVITY & ABLATION
**Investigation**: Which hyperparameters are critical, and how do they interact?

**What we study**:
- One-Factor-At-a-Time (OFAT) sensitivity analysis
- Interaction effects between hyperparameters (e.g., learning rate × discount factor)
- Hard thresholds vs. smooth trade-offs in parameter space
- Joint optimization (Bayesian optimization)
- Robustness of baseline configurations

---

### PHASE 6: DEEP REINFORCEMENT LEARNING
**Investigation**: How do neural network-based methods compare to tabular approaches?

**What we study**:
- DQN with experience replay and target networks
- DDQN with bias reduction
- Comparison vs. tabular baseline
- Sample efficiency and convergence speed
- Policy smoothness and generalization
- Learned value function structure

---

### PHASE 7: CONTINUOUS CONTROL METHODS
**Investigation**: How do different policy gradient methods handle continuous action spaces?

**What we study**:
- DDPG (deterministic policy gradient with actor-critic)
- REINFORCE (vanilla policy gradient with baseline)
- A2C (advantage actor-critic)
- Performance on dynamics-complex problems
- Performance on reward-misaligned problems
- Deterministic vs. stochastic policy trade-offs

---

## RESEARCH QUESTIONS

### Representation & Scaling
- How does state discretization resolution affect learning efficiency and performance?
- When and why do neural network methods outperform tabular methods?
- How does continuous state representation impact performance compared to discretized states?

### Algorithm Selection
- Which tabular algorithm is most suitable for sparse-reward long-horizon control?
- What are the trade-offs between off-policy and on-policy learning?
- Under what conditions do certain algorithms fail completely?

### Hyperparameter Tuning
- Which hyperparameters are critical vs. relatively robust to variation?
- Are there hard thresholds or smooth performance curves in parameter space?
- How do hyperparameters interact, and can we optimize them jointly?

### Reward Design
- How does reward shaping affect learned behavior and convergence difficulty?
- Can advanced algorithms overcome fundamentally misaligned reward signals?
- What makes a reward signal learnable vs. unlearnable?

### Control Methods
- When should deterministic policies be preferred over stochastic policies?
- How do continuous action methods compare to discrete action approximations?
- What is the efficiency trade-off between different policy gradient methods?

---

## EXPERIMENTAL METHODOLOGY

- **Multiple seeds**: Robustness assessment across random initializations
- **Fixed evaluation budgets**: Fair comparison between methods with similar computational resources
- **Standardized metrics**: Success rate, convergence speed, sample efficiency, policy quality
- **Visualization-first analysis**: Phase portraits, policy heatmaps, value landscapes, learning curves
- **Ablation studies**: Isolate effects of individual variables before combining
- **Parallelization**: Efficient batch training across configurations

---

## CROSS-CUTTING THEMES

1. **Representation Impact**: State space encoding (continuous, discretized, learned) and scalability
2. **Algorithm Suitability**: Which algorithms fit which problem structures
3. **Hyperparameter Criticality**: Sensitivity analysis and parameter interaction
4. **Scalability**: How methods scale from tabular to neural approaches
5. **Robustness**: Consistency across seeds and configurations
6. **Algorithm vs. Reward Design**: Relative importance of method sophistication vs. reward alignment

---

## INFRASTRUCTURE

- **Environment Wrapper**: Unified interface for discrete/continuous variants with multiple reward scenarios
- **Agent Library**: Modular implementations of tabular, deep RL, and policy gradient methods
- **Evaluation Pipeline**: Standardized training, metrics collection, statistical analysis
- **Visualization Suite**: Learning curves, policies, trajectories, value functions
- **Experiment Tracking**: Results export, agent state persistence

---
