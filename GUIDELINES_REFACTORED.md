# 🏔️ Mountain Car RL Study - REFACTORED GUIDELINES (Option A)
# Sequential & Coherent Phase Structure

---

## PROBLEM OVERVIEW

**Goal**: Comprehensive RL study of Mountain Car under different conditions, solving the problem with various methods and analyzing trade-offs.

**Key Constraints**:
- Continuous state space: position [-1.2, 0.6], velocity [-0.07, 0.07]
- Must learn momentum strategy (non-obvious solution)
- Different RL families have different strengths
- Reward engineering dramatically affects learning

---

## PHASE STRUCTURE (SEQUENTIAL & CONSEQUENTIAL)

### PHASE 0: PROJECT INFRASTRUCTURE ✅ DONE
**Deliverable**: `src/` module structure

- ✅ `src/environment_utils.py` - Environments + reward wrappers
- ✅ `src/evaluation.py` - Generic training loop + metrics
- ✅ `src/agents/` - Agent implementations
- ✅ `src/plotting.py` - Standardized visualizations

---

### PHASE 1: ENVIRONMENT & BASELINE UNDERSTANDING ✅ DONE
**Deliverable**: `notebooks/01_environment_and_baselines.ipynb`

**Why Phase 1?**: Establish foundation, understand problem.

**Content**:
- Environment inspection (state/action spaces)
- State discretization mechanics (20×20 bins baseline)
- Q-learning baseline (single seed, understand algorithm)
- Training loop + evaluation infrastructure
- Learning curves, policy heatmaps, trajectories
- TensorBoard logging setup

**Questions answered**:
- How does the Mountain Car work?
- Can Q-learning solve it with discrete actions?
- What does a learned policy look like?

**Outcomes**:
- Baseline performance metrics (discrete)
- Trained Q-table for reference
- Pipeline for training/evaluation ready

---

### PHASE 2: STATE REPRESENTATION & DISCRETIZATION ANALYSIS ✅ DONE
**Deliverable**: `notebooks/02_discretization_analysis.ipynb`

**Why Phase 2?**: Before comparing algorithms, understand representation impact.

**Key Insight**: Discretization is a critical design choice. How many bins?

**Content**:
- Test bin counts: {5, 10, 15, 20, 30, 40}
- Train Q-learning with each configuration (same hyperparams)
- Track: convergence speed, final performance, stability, computation time
- Visualize: learning curves overlay, policy heatmaps comparison
- Measure: "sweet spot" for bins (performance vs efficiency)

**Experiments**:
- Environment: Discrete Mountain Car (MountainCar-v0)
- Scenario: min-steps only
- Same agent hyperparams (α=0.1, γ=0.99)
- 3-5 seeds per config for robustness
- Comparison table: Bins vs Performance

**Questions answered**:
- How many bins do we actually need?
- Diminishing returns at what point?
- Can coarse discretization still learn?

**Outcomes**:
- ✅ **Optimal discretization identified: 10×10 bins (100 states)**
- Understanding of representation-performance trade-off
- Key finding: finer discretizations (20×20+) actually hurt performance (overfitting)

---

### PHASE 3: REWARD ENGINEERING & SCENARIO DESIGN (DISCRETE ONLY)
**Deliverable**: `notebooks/03_reward_scenarios.ipynb`

**Why Phase 3?**: After fixing representation, explore reward design.

**Key Insight**: Different rewards optimize for different goals.

**Scenarios** (DISCRETE ONLY):

| Scenario | Reward | Goal | Challenge |
|----------|--------|------|-----------|
| **Min-Steps** | -1/step | Speed | Pure performance |
| **Min-Fuel** | -1/step - 0.25·thrust | Efficiency | Trade-off: speed vs actions |

**Content**:
- Train Q-learning on min-steps (baseline from Phase 1)
- Train Q-learning on min-fuel (new scenario)
- Compare learned policies side-by-side
- Analyze behavioral differences:
  - Action frequency (how many thrust actions?)
  - Path efficiency (direct vs roundabout?)
  - Convergence patterns
- Measure:
  - Steps to goal
  - Actions taken
  - Success rate
  - Fuel efficiency

**Experiments**:
- Environment: Discrete MountainCar-v0 only
- Discretization: **10×10 bins** (optimal from Phase 2)
- Algorithm: Q-learning (proven from Phases 1-2)
- Multi-seed: 5 seeds per scenario
- Hyperparams: α=0.1, γ=0.99 (standard)

**Questions answered**:
- How does reward shape affect learned behavior?
- Min-steps vs min-fuel: which is easier to learn?
- Can we predict the reward from the learned policy?
- What's the efficiency-speed trade-off?

**Outcomes**:
- Reward design impact quantified
- Policies for different discrete objectives
- Understanding of how agents adapt to reward structure

**Note on Continuous**: Continuous scenarios (linear cost, quadratic cost) are **deferred to Phase 7** (Policy Gradient Methods) because they require continuous action handling, not discrete Q-learning.

---

### PHASE 4: TABULAR METHODS COMPARISON
**Deliverable**: `notebooks/04_tabular_methods_comparison.ipynb`

**Why Phase 4?**: Now compare algorithms on fixed representation + reward.

**Key Insight**: Off-policy vs on-policy has different trade-offs.

**Methods**:
1. **Q-Learning** (off-policy) - learns optimal policy
2. **SARSA** (on-policy) - respects exploration
3. **Monte Carlo** (returns-based) - high variance

**Content**:
- Train all 3 methods on min-steps scenario
- Multi-seed runs (5 seeds minimum)
- Hyperparameter grids:
  - α ∈ {0.05, 0.1, 0.2}
  - γ ∈ {0.95, 0.99}
  - ε_decay ∈ {0.995, 0.999}
- Comparison table: performance, convergence, stability
- Learning curves for each method
- Policy visualization comparison

**Questions answered**:
- Which method converges fastest?
- Which is most stable across seeds?
- Does off-policy really outperform on-policy?
- Monte Carlo vs TD: which is better for this problem?
- Which method has highest sample efficiency?

**Outcomes**:
- Best tabular method identified
- Robustness analysis (seed variance)
- Hyperparameter effects understood

---

### PHASE 5: HYPERPARAMETER SENSITIVITY & ABLATION
**Deliverable**: `notebooks/05_hyperparameter_sensitivity.ipynb`

**Why Phase 5?**: Understand which hyperparams matter most.

**Content**:
- Systematic ablation studies:
  - Learning rate effect (α: 0.01, 0.05, 0.1, 0.2, 0.5)
  - Discount factor effect (γ: 0.9, 0.95, 0.99, 0.999)
  - Exploration decay rates (ε_decay: 0.99, 0.995, 0.999, 0.9995)
  - Initial epsilon vs min epsilon
- One-factor-at-a-time experiments
- Sensitivity rankings (which param matters most?)
- Interaction effects (α vs γ trade-off)
- Visualization: sensitivity heatmaps

**Experiments**:
- Use best method from Phase 4
- Fixed other params to Phase 4 optimal
- Vary one param at a time
- 3-5 seeds per config
- Plot: sensitivity curves + heatmaps

**Questions answered**:
- Which hyperparameter has biggest impact?
- Are there critical thresholds?
- How robust is the solution to param variation?
- What's the "magic" hyperparameter set?

**Outcomes**:
- Hyperparameter sensitivity map
- Robust vs brittle params identified
- Guidelines for hyperparameter selection

---

### PHASE 6: SCALING TO LARGER STATE SPACES (Deep RL)
**Deliverable**: `notebooks/06_deep_rl_networks.ipynb`

**Why Phase 6?**: Tabular methods scale poorly to high-dim spaces. Use neural networks.

**Key Insight**: Function approximation via neural networks enables scaling.

**Methods**:
- **DQN**: Experience replay + target network (standard)
- **DDQN**: Double Q-network (reduces overestimation)

**Content**:
- Network architecture: 
  - Input: [position, velocity] (continuous state)
  - Hidden: Dense(64) ReLU → Dense(64) ReLU
  - Output: Q-values for 3 discrete actions
- Experience replay buffer (capacity: 10000)
- Target network with soft updates (τ=0.001)
- Training on min-steps scenario (discrete actions)
- Comparison vs best tabular method from Phase 4

**Benchmarks**:
- Convergence speed (episodes to 80% success)
- Final performance (eval reward)
- Stability (seed variance)
- Training time per episode
- Q-value landscapes

**Experiments**:
- Environment: Discrete MountainCar-v0 (same as Phases 1-5)
- Action discretization: keep 3 actions {-1, 0, 1}
- Multi-seed: 5 seeds
- Batch size: 32, 64
- Learning rate: 0.0001, 0.001
- Comparison table: Tabular vs DQN vs DDQN

**Questions answered**:
- Does neural network improve over Q-learning?
- What's the trade-off: performance vs interpretability?
- When should you use DQN vs Q-learning?
- How much does DDQN help?

**Outcomes**:
- Deep RL baseline established
- Understanding of function approximation benefits/costs
- Foundation for Phase 7 continuous control

---

### PHASE 7: CONTINUOUS CONTROL (Policy Gradient Methods)
**Deliverable**: `notebooks/07_continuous_control.ipynb`

**Why Phase 7?**: Discrete methods can't handle continuous actions. Policy gradient naturally handles them.

**Key Insight**: Policy gradient directly outputs action distribution (no Q-table needed).

**Environments**:
- **MountainCar-Continuous-v0**: Continuous actions in [-1.0, 1.0]

**Reward Scenarios** (from Phase 3, adapted for continuous):

| Scenario | Reward | Goal |
|----------|--------|------|
| **Linear Cost** | -0.1·\|action\| | Smooth control |
| **Quadratic Cost** | -0.1·action² | Energy efficiency |

**Methods**:

1. **REINFORCE** (Vanilla Policy Gradient)
   - Architecture: π(a\|s) → μ(s), σ(s) (Gaussian distribution)
   - Direct policy optimization
   - High variance but unbiased

2. **Actor-Critic** (Policy + Value Network)
   - Actor: π(a\|s) → action
   - Critic: V(s) → state value
   - Lower variance, biased
   - Faster convergence

**Content**:

- Network architectures:
  - Actor: [pos, vel] → Dense(64) → Dense(64) → [μ, log_σ]
  - Critic: [pos, vel] → Dense(64) → Dense(64) → [V(s)]
  
- Training on both linear and quadratic reward scenarios
- Multi-seed runs (5 seeds)
- Comparison:
  - REINFORCE vs Actor-Critic
  - Linear vs Quadratic rewards
  - DQN (from Phase 6) discretized actions vs continuous policy

**Metrics**:
- Convergence speed
- Final performance
- Action smoothness (std of actions)
- Reward accumulated
- Success rate

**Experiments**:
- Environment: MountainCar-Continuous-v0
- Action space: continuous [-1.0, 1.0]
- Multi-seed: 5 seeds
- Learning rate: 0.0001, 0.001, 0.01
- Epochs: 5000

**Questions answered**:
- Is continuous control really different?
- Which method is better: REINFORCE or Actor-Critic?
- Linear vs Quadratic: which reward is harder?
- Comparison: how do discrete actions (DQN) vs continuous compare?
- Can we learn smooth policies?

**Outcomes**:
- Continuous control solutions for multiple reward scenarios
- Method selection guide (when to use REINFORCE vs Actor-Critic)
- Understanding of policy gradient advantages

---

### PHASE 8: POLICY ANALYSIS & INTERPRETATION
**Deliverable**: `notebooks/08_policy_interpretation.ipynb`

**Why Phase 8?**: After solving, understand solutions deeply.

**Key Insight**: Good ML is interpretable ML.

**Content**:

**Numerical Analysis**:
- Policy heatmaps for all methods (tabular)
- Policy smoothness metrics (continuous)
- State coverage analysis
- Q-value landscapes (tabular)

**Physical Interpretation**:
- Does policy make physical sense?
- Momentum exploitation?
- Efficiency patterns?
- Generalization to different starting positions?

**Comparative Analysis**:
- Policy similarity across methods (distance metrics)
- Policy robustness to perturbations
- Do different methods find similar solutions?

**Statistical Analysis**:
- Q-value distributions
- Value function uncertainty
- Confidence intervals on policies
- Variance across seeds

**Explainability**:
- Feature importance (position vs velocity)
- Decision trees approximating learned policies
- Rule extraction ("IF pos > X AND vel > Y THEN push left")
- Saliency maps for neural network policies

**Questions answered**:
- Why does each learned policy work?
- Are different methods finding similar solutions?
- Can we extract interpretable rules from neural nets?
- Which policies are robust?

**Outcomes**:
- Deep understanding of learned solutions
- Interpretability score for each method
- Guidelines for policy deployment

---

### PHASE 9: SYNTHESIS & COMPREHENSIVE COMPARATIVE ANALYSIS
**Deliverable**: `notebooks/09_synthesis_and_conclusions.ipynb`

**Why Phase 9?**: Synthesize all findings into actionable insights.

**Content**:

**Master Comparison Table**: All methods × all scenarios

| Method | Type | Scenario | Converge | Performance | Stability | Interpretable | Continuous | Best For |
|--------|------|----------|----------|-------------|-----------|---------------|-----------|----------|
| Q-learn | Tabular | Min-steps | Fast | ⭐⭐⭐⭐ | High | ✅ | ❌ | Small discrete |
| SARSA | Tabular | Min-steps | Medium | ⭐⭐⭐ | V.High | ✅ | ❌ | Safe discrete |
| MC | Tabular | Min-steps | Slow | ⭐⭐⭐ | Medium | ✅ | ❌ | Exact returns |
| DQN | Deep | Min-steps | Fast | ⭐⭐⭐⭐⭐ | Medium | ❌ | ❌ | Scale discrete |
| DDQN | Deep | Min-steps | Fast | ⭐⭐⭐⭐⭐ | Medium-High | ❌ | ❌ | Stable scaling |
| REINFORCE | Policy | Continuous | Slow | ⭐⭐⭐ | Low | Medium | ✅ | Simple continuous |
| Actor-Critic | Policy | Continuous | Fast | ⭐⭐⭐⭐ | High | Medium | ✅ | Production continuous |

**Key Findings**:
1. Best method for each problem type
2. Performance vs interpretability trade-off
3. Computational requirements ranking
4. Sample efficiency comparison
5. Robustness to hyperparameters
6. Continuous vs Discrete differences

**Research Questions Answered**:
- Which RL family is best for Mountain Car?
- Does discretization granularity matter? (Phase 2)
- Does reward design affect learning? (Phase 3)
- Which tabular method wins? (Phase 4)
- Which hyperparameters matter most? (Phase 5)
- Does deep RL help? (Phase 6)
- How different is continuous control? (Phase 7)
- Can we extract interpretable policies? (Phase 8)

**Practical Guidelines**:
- **For production**: Use Actor-Critic (continuous) or DQN (discrete large)
- **For understanding**: Use tabular Q-learning + policy analysis
- **For speed**: Use DQN or Actor-Critic
- **For safety/robustness**: Use SARSA (discrete) or Actor-Critic (continuous)
- **For interpretability**: Use tabular methods + rule extraction

**Limitations & Future Work**:
- What we didn't explore
- Extensions (multi-task, transfer learning, meta-learning, inverse RL)
- Theoretical insights
- Open problems

**Outcomes**:
- Complete RL study of Mountain Car
- Actionable method selection guide
- Framework for future RL projects
- Insights into RL design choices

---

## SUMMARY: WHY THIS ORDER? (Option A)

```
Phase 1: Understand the problem (discrete actions)
    ↓
Phase 2: Optimize representation (how many bins?)
    ↓
Phase 3: Explore reward design (discrete scenarios only)
    ↓
Phase 4: Compare base algorithms (tabular: Q vs SARSA vs MC)
    ↓
Phase 5: Understand algorithm parameters (ablation)
    ↓
Phase 6: Scale with deep RL (DQN/DDQN for larger spaces)
    ↓
Phase 7: Handle continuous control (Policy Gradient for continuous actions)
    ↓
Phase 8: Interpret & explain (why do policies work?)
    ↓
Phase 9: Synthesize & recommend (what did we learn?)
```

**Key Features of Option A**:
- ✅ **No jumping back**: Continuous only handled in Phase 7 (natural place)
- ✅ **Logical progression**: Discrete foundation first, then scale and extend
- ✅ **No method gaps**: Each phase uses appropriate methods
- ✅ **Pedagogical**: Learn tabular RL deeply before neural networks

---

## GRADING BREAKDOWN

- **Phase 1**: Foundation (10%)
- **Phase 2-3**: Problem analysis & representation (15%)
- **Phase 4-5**: Algorithm comparison & tuning (20%)
- **Phase 6**: Scaling methods (15%)
- **Phase 7**: Continuous control (15%)
- **Phase 8-9**: Interpretation & synthesis (25%)

---

## TESTING & DEPLOYMENT

- ✅ All notebooks runnable (reproducibility)
- ✅ TensorBoard logging throughout
- ✅ Multi-seed robustness (3-5 seeds per experiment)
- ✅ Statistical significance testing
- ✅ Code quality (no warnings)
- ✅ Documentation complete

