# Mountain-Car


## Introduction

### Project context

This project studies how different reinforcement learning (RL) algorithms behave on the **Mountain Car** control problem. The goal is not only to make an agent reach the target, but also to understand **how different RL families learn**, how their behavior changes under different design choices, and which methods are most suitable for different versions of the same problem.

Mountain Car is a classic RL benchmark because it is simple to describe but surprisingly rich from a learning perspective. A small underpowered car starts in a valley and must reach the top of a hill. The car cannot drive straight to the goal because its engine is too weak. To succeed, it must first move away from the target, build momentum by swinging back and forth, and only then use that momentum to climb the hill. This makes the task a good example of **sequential decision-making**, where a short-term action that looks unhelpful may actually be necessary for long-term success.

The environment is especially useful for comparing RL methods because it combines a relatively small state description with meaningful control difficulty.

The state is defined by two continuous variables:

* the car's **position**
* the car's **velocity**

Depending on the version of the environment, the action space can be:

* **discrete**, where the agent chooses between pushing left, not pushing, or pushing right
* **continuous**, where the agent can choose the strength of the applied force

This makes Mountain Car a strong testbed for comparing:

* **tabular value-based methods** such as Monte Carlo, SARSA, and Q-learning
* **deep value-based methods** such as DQN and DDQN
* **policy-based methods** such as policy gradient approaches

In other words, the same conceptual task can be studied under multiple RL formulations, which is ideal for a comparative assignment.

### Main challenge

The agent must learn a strategy that is **physically unintuitive at first**. A naive controller might always push toward the goal, but that fails because the engine alone is not strong enough to climb the hill directly. The agent must instead learn to exploit the system dynamics, using velocity gained from previous movements to reach the objective.

This means success depends on learning:

* how current actions affect future states
* how to balance immediate reward against long-term payoff
* how to explore enough to discover the swinging strategy

Because of this, Mountain Car is a useful environment for analyzing how different algorithms handle exploration, delayed reward, and long-term planning.

### Project objective

The objective of this project is to build, train, evaluate, and compare several reinforcement learning algorithms on Mountain Car and its variants. We are not only interested in which method performs best numerically, but also in:

* how stable training is
* how sensitive each method is to hyperparameters
* how state and action representations affect performance
* how the learned policies can be interpreted visually and conceptually

A first major distinction in this project is between methods that require a **tabular state representation** and methods that can work directly with the raw continuous state.

For example, classical methods like Q-learning and SARSA use Q-tables. Since Mountain Car states are continuous, we cannot directly assign one table entry to every exact state value. For these methods, we therefore approximate the state space through **discretization**, dividing position and velocity into bins and treating each bin combination as a discrete state. This introduces an important design trade-off:

* a coarse discretization is simpler and learns faster
* a fine discretization is more precise but requires more experience

This trade-off is itself part of the analysis.

### Experimental plan

The project is structured progressively.

We begin with the **discrete-action environment** and implement strong baseline methods, starting with tabular approaches such as Q-learning and SARSA. This lets us establish a full experimental pipeline:

* environment inspection
* state discretization
* training
* evaluation
* reward monitoring
* policy visualization

Once the tabular baseline is understood, we extend the comparison to more advanced methods such as Monte Carlo, DQN, and DDQN. Finally, for the continuous-action version of the problem, we use a policy-based method that is more natural for continuous control.

This progression is intentional. It allows the project to move from simple and interpretable methods toward more flexible and powerful ones, while keeping the comparison grounded.

And by the end of the project, we want to answer questions such as:

* Which RL family is most effective for the standard Mountain Car task?
* How much does discretization affect tabular performance?
* Do deep value-based methods provide a clear advantage over tabular ones?
* How do reward design and environment variation influence the learned strategy?
* Are the best numerical policies also easy to interpret physically?
