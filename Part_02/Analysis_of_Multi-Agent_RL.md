REINFORCEMENT LEARNING INTRODUCTION
Assignment 22.00 — Group Assignment | Part 02

---

# A Critical Analysis of Multi-Agent Reinforcement Learning

for Dynamic Pricing in Supply Chains

## Abstract

This paper analyzes the application of Multi-Agent Reinforcement Learning (MARL) to dynamic pricing in supply chains, as studied by Hazenberg et al (2025) together with KPMG N.V. and the University of Amsterdam. The study is analyzed as an outstanding example of applied RL for its methodological sophistication: it justifies the simulation environment in over one million real retail transactions, builds a data-driven demand model using LightGBM, and evaluates three modern MARL algorithms, MADDPG, MADDQN, and QMIX, opposed to static rule-based baselines. The analysis shows a fundamental tension at the center of competitive dynamic pricing: more adaptive agents generate higher revenue but produce less fair price distributions across the market. For the purpose of analytical depth, this finding is interpreted against the 2018 BBVA study on single-agent fair pricing, serving as a methodological baseline. Together, the two works demonstrate a critical evolution in the field: from designing fairness explicitly into a reward function, to understanding how fairness emerges, or fails to, from competitive multi-agent dynamics.

## I. Introduction

Dynamic pricing is the practice of adjusting prices in real time based on demand, competition, stock, and customer behavior; being one of the most commercially significant applications of machine learning today. Airlines reprice seats millions of times a day, on-demand transportation platforms like Uber implement surge pricing within seconds, E-commerce leaders such as Amazon update product prices hundreds of millions of times daily. Under this operational reality lies a really difficult decision making problem: the challenge of how a seller should set prices over time, given uncertainty about demand, competitors strategies, and the long-term consequences of today’s choices.

Reinforcement Learning offers a compelling structure for this problem. Unlike static optimization or rule-based systems, RL agents learn from sequential interaction with the environment, balancing the exploration of new pricing strategies against the exploitation of known profitable ones. The temporal structure of the reward, where today’s prices affect tomorrow’s demand, outlines naturally onto the Markov Decision Process that underlies most RL methods.

The field’s early work on RL for dynamic pricing, demonstrated by Maestre et al. (2018) at BBVA Data & Analytics in Madrid, operated under a critical simplifying assumption: a single agent prices for a population of customers in isolation. This is analytically clean but economically incomplete. Real markets are competitive: when a retailer changes prices, competitors react. When a supplier adjusts margins, downstream distributors adapt. These relationships are not noise that should be filtered out; they are the mechanism by which markets function.

Hazenberg et al. (2025), in a collaboration between the University of Amsterdam and KPMG N.V., take a step toward this more realistic formulation. Introducing Multi-Agent Reinforcement Learning (MARL) to the supply chain pricing problem, banking what happens when multiple autonomous agents, each representing a seller with its own product listing, learn pricing strategies simultaneously in a shared, competitive market. Resulting in a methodologically enriching study, analytically more complex than anything that had been explored in this specific domain.

This paper serves a structured analysis of that study: its motivation and suitability as an RL application, the design of its methodology and MDP formulation, the interpretation of its results, and a personal critical evaluation of its contribution and limitations. Where relevant, comparisons to the BBVA baseline are drawn to highlight how the field has evolved since 2018.

## II. Problem Identification and Solution Suitability

Dynamic pricing in competitive supply chains shows the characteristics of a problem that make RL not only a suitable solution but the appropriate solution. The problem is naturally sequential; that is, the decision taken at a particular point of time changes demand, inventory levels, and competitors’ behavior, which influence future decisions. The environment is non-stationary, since the agent’s strategy is a part of the world experienced for other agents as well. Thus, there are no stationary states as per classical analysis. There is also a delay in rewards, demand is unpredictable, and no closed-form optimal policy exists.

The competitive structure of supply chain markets then suggests specifically toward its multi-agent variant. In the BBVA paper published in 2018, a single agent determined the pricing for a specific set of customers whose behaviour pattern was fixed; the environment did not adapt to the agent. In Hazenberg et al.(2025), there is an interaction between the agents, which means that the environment itself is made up of another set of learning agents. When a retailer changes prices, the rival reacts to this change and the reaction become a part of the state of the initial agent. Modelling the interdependence explicitly, as MARL does, is the difference between solving the problem that exists on paper and the one that exists in practice

## III. Methodology

### Data Foundation and Demand Modelling

A central methodological strength of this study is its grounding in real transactional data. The Online Retail II dataset from the UCI Machine Learning Repository contains over one million transactions from a UK-based giftware retailer operating between 2009 and 2011, covering 5243 unique products and 5876 customers. This is not a synthetic pricing environment, the statistical properties of demand, seasonality, and pricing sensitivity are derived from observed behaviour.

Rather than feeding raw transactions directly to RL agents (which would make the simulation computationally interactable and dependent on replying historical data), the authors train a LightGBM gradient boosting model to predict weekly demand as a function of price and contextual features. The model yielded on the held-out test set, a level of predictive accuracy sufficient to serve as a reasonable surrogate for real market behaviour.

The feature engineering pipeline is notably detailed; temporal indicators (week, weekday, holiday flags), semantic product clusters derived from Sentence-BERT embeddings, rolling demand averages, prices elasticity indicators, and interaction terms between price and seasonality. This richness means the model captures non-linear and context-dependent price sensitivity, rather than assuming a fixed elasticity. The estimated average price elasticity confirms that giftware is highly inelastic, a finding consistent with domain knowledge and with intuition that premium gifts are relatively insensitive to moderate price changes.

One critical observation is that LightGBM’s demand predictions function as the entire environment for the RL agents. This means that any systematic bias or out-of-distribution extrapolation in the demand model propagates directly into the agent's learned policies. The authors acknowledge this limitation, but do not conduct sensitivity analysis to quantify its impact, a gap that weakened the study empirical claims.

### Simulation Environment Design 

The simulation environment operated on a weekly time step: at each step, agents observe market conditions (week number, holiday indicator, product category clusters), set prices for their product portfolios , and receive demand predictions and resulting revenue as feedback. The environment maintains a centralized history log of all agent actions and outcomes, which is used for centralized training in the CTDE (Centralized Training, Decentralized Execution) paradigm. 

The weekly time step is a paradigm choice given the dataset’s temporal resolution, but it represents a significant departure from operational reality. Real dynamic pricing systems, particularly in e-commerce and ride-hailing, operate on minute or second time scales. A weekly agent may miss intraday demand peaks, real-time competitor responses, and flash pricing events that constitute much of the strategic richness in real markets. This temporal abstraction is a limitation worth flagging explicitly in any analysis.

### MDP Formulation
State Space 
The state observed by each agent at each time step consists of: the current week number, a binary holiday indicator, the product category cluster assignment, previous-week sales, and a rolling demand history. Notably, agents also observe a market summary, a shared dictionary of conditions passed by the environment, enabling partial awareness of competitive context without full observability of competitors' pricing decisions.  
The state design represents a meaningful advance over the BBVA formulation, which encoded only two variables: the customer group identity and the current Jain fairness index. The 2025 state space is richer, more temporally structured, and grounded in features that are genuinely causally relevant to demand. However, it remains partially observable in a theoretically important sense: agents do not directly observe competitors' prices or inventory levels, only the aggregate demand outcome they receive. This is realistic, in real markets, competitor prices must be inferred from observed demand shifts, but it also means agents are learning under significant information asymmetry.

Action Space 
Each agent's action is a continuous pricing decision for each product in its portfolio. The action space is bounded by a minimum and maximum price range derived from the dataset's price distribution. MADDPG and QMIX operate on continuous action representations; MADQN discretizes the price range into a finite set of levels, trading representational fidelity for computational tractability.
This distinction between continuous and discrete action spaces is analytically significant. Continuous pricing allows for fine-grained strategic behaviour, matching a competitor's price exactly, or pricing 2% below cost to capture market share. Discrete pricing rounds to the nearest level, introducing a granularity constraint that may systematically advantage or disadvantage certain strategies. The paper does not analyse this effect directly, which is a missed opportunity.

Reward Function
The reward for each agent at each time step is its total revenue: the sum of (price × predicted demand) across all products in its portfolio. This is a pure profit motive with no fairness term, a deliberate departure from the BBVA reward design, which explicitly encoded a fairness objective in the reward function.
This design choice has profound implications for the results. In the BBVA study, fairness was a first-class citizen of the optimization objective, the agent was literally rewarded for treating customer groups equally. In the 2025 study, fairness is entirely absent from individual agents' reward signals. Any fairness that emerges in the results is not designed, it is a consequence of the competitive equilibrium the agents reach. This makes the results far more interesting analytically: they tell us what fairness looks like when nobody is trying to achieve it, but competitive pressure forces some degree of price alignment.

###Algorithm Selection and Architecture  
The reward for each agent at each time step is its total revenue: the sum of (price × predicted demand) across all products in its portfolio. This is a pure profit motive with no fairness term, a deliberate departure from the BBVA reward design, which explicitly encoded a fairness objective in the reward function.
This design choice has profound implications for the results. In the BBVA study, fairness was a first-class citizen of the optimization objective, the agent was literally rewarded for treating customer groups equally. In the 2025 study, fairness is entirely absent from individual agents' reward signals. Any fairness that emerges in the results is not designed, it is a consequence of the competitive equilibrium the agents reach. This makes the results far more interesting analytically: they tell us what fairness looks like when nobody is trying to achieve it, but competitive pressure forces some degree of price alignment.
 
The selection of these three algorithms is well-motivated: they span the key design dimensions of MARL (value-based vs. policy gradient, discrete vs. continuous actions, cooperative vs. competitive training signals). The inclusion of rule-based baselines, which set prices according to fixed heuristics such as undercutting the lowest competitor price, provides an important reference point for evaluating whether RL's complexity is justified by its performance.


IV.     Results and Interpretation

Comparative Performance

The study evaluates each agent configuration across four dimensions: average profit (revenue generated), price stability (measured as coefficient of variation across time), market fairness (Jain's Index applied to market share distribution), and market competitiveness (share volatility across agents). The results reveal a fundamental tradeoff between adaptability and fairness that runs through every comparison.

| Agent/Strategy | Avg. Revenue  | Price Stability | Fairness (Jain’s Index) | Market Competition |
| -------------- | ------------- | --------------- | ----------------------- | ------------------ |
| Rule-Based     | Baseline      | Highest (0.024) | Near-perfect (0.9896)   | None (static)      |
| MADDPG         | Moderate-High | Moderate        | High (0.8819)           | Moderate (9.5 pp)  |
| QMIX           | Moderate      | Moderate-High   | Moderate-High           | Low-Moderate       |
| MADQN          | Highest       | Lowest          | Lowest (0.54844)        | Highest            |

Table 1. Summary of agent performance across key evaluation dimensions. Source: Hazenberg et al. (2025).

The Adaptability-Fairness Tradeoff

The most analytically striking finding is that the most capable learning agent is also the least fair. MADQN, which achieves the highest revenue, produces a Jain's Index of just 0.5844, meaning price distributions across market segments are highly unequal. By contrast, the static rule-based agent achieves near-perfect fairness (0.9896) precisely because it never adapts; equal treatment is the default when no agent is strategically differentiating

This result inverts the intuition that intelligence should produce better social outcomes. The mechanism is clear once examined: MADQN agents learn to aggressively undercut competitors in high-demand periods and surge in low-competition windows, creating systematic price discrimination across time and customer segment. No single agent intends to be unfair, but the competitive equilibrium they reach collectively produces unfair market outcomes.

MADDPG presents a more balanced profile: it achieves competitive revenue with a fairness score of 0.8819, suggesting that the actor-critic architecture with centralized training encourages more stable and less predatory pricing strategies. This may be because the centralized critic implicitly models the consequences of extreme price moves, when all agents are visible during training, highly aggressive strategies are countered and thus less reinforced.

The Emergent vs. Designed Fairness Comparison

This result becomes most meaningful when placed against the BBVA baseline. Maestre et al. (2018) demonstrated that a single RL agent can achieve fairness scores of up to 99%,  but only by explicitly including a fairness term (the rotated Jain's Index) in its reward function. Fairness was a design choice, a hyperparameter-controlled objective that the agent was directly incentivized to pursue.

In the 2025 study, no agent receives any fairness incentive. The 88% fairness achieved by MADDPG is entirely emergent; it arises from the competitive equilibrium, not from any normative constraint in the reward signal. This distinction is critical for practitioners: if you want guaranteed fairness in a multi-agent pricing system, you cannot rely on competition alone. You must either engineer fairness into the reward (as BBVA did) or impose regulatory constraints externally.

The 58% fairness of MADQN represents a market failure in the economic sense: a configuration that is individually rational for each agent but collectively suboptimal from a societal perspective. This is precisely the kind of outcome that algorithmic pricing regulation, currently being debated in the EU AI Act and related frameworks, is designed to address.

Methodological Quality of Result Analysis

The study evaluates results across multiple random seeds, providing variance estimates alongside mean performance figures. This is methodologically sound and underappreciated in applied RL research, where single-run results are too often reported as representative. The stability analysis reveals that MAPPO (in the comparative literature) achieves lower variance than MASAC, confirming that not all high-performing agents are equally reproducible, a practically important distinction for deployment.

The use of Jain's Index as a fairness metric is consistent with the BBVA baseline, enabling direct comparison. However, the study does not decompose fairness by customer segment, product category, or time period, all of which would be analytically valuable. An agent that achieves 88% average fairness might still be systematically unfair to a specific customer group or during holiday periods.

V.      Critical Personal Evaluation

Strengths

Real-world data grounding: The use of the UCI Online Retail II dataset distinguishes this study from the majority of MARL pricing research, which relies on fully synthetic environments. The LightGBM demand model creates a bridge between observed customer behaviour and the simulation, lending credibility to the results that pure simulation studies cannot claim.

Algorithmic breadth: By benchmarking three fundamentally different MARL architectures alongside rule-based baselines, the study provides genuine comparative insight rather than advocating for a single approach. The finding that different algorithms produce qualitatively different market behaviours is itself a significant result.

Multi-dimensional evaluation: The simultaneous measurement of profit, stability, fairness, and competitiveness reflects a mature understanding of what it means for a pricing system to perform well. Revenue maximization alone is an insufficient criterion for real-world deployment.

Connection to practice: The collaboration with KPMG N.V. ensures that the research questions are framed with industrial relevance in mind, and the discussion of ERP integration provides a plausible deployment pathway.

Limitations and Open Questions

Domain specificity: The giftware sector has an estimated price elasticity of −0.072, highly inelastic. The algorithms' behaviour in elastic markets (e.g., travel, consumer electronics) may differ substantially. Findings should not be generalized across sectors without further validation.

Temporal abstraction: Weekly time steps abstract away the intraday dynamics that constitute most of the strategic richness in real dynamic pricing. The competitive landscape at 9am on a Thursday is different from 4pm on a Sunday, and an agent operating at weekly resolution cannot learn to exploit these patterns.

No collusion testing: The study does not test whether MARL agents converge to implicit collusion, a known risk in algorithmic pricing where agents independently learn to maintain supracompetitive prices. This is not merely an academic concern; the European Commission has investigated algorithmic pricing collusion in multiple sectors. A rigorous study of real-world applicability should address this question.

Limited fairness decomposition: Reporting aggregate Jain's Index scores conceals within-group variation. A 0.88 average could mask severe unfairness toward specific segments. Given the study's explicit interest in fairness, this is a significant analytical gap.

Absence of online learning: The policies are trained offline and then evaluated. Real pricing systems must update continuously as market conditions evolve. The study does not address how these MARL policies would behave in continual learning settings, a crucial consideration for operational deployment.

Personal Assessment of Suitability

The selection of MARL for this problem is, in this author's assessment, the correct choice, not merely adequate, but demonstrably superior to the single-agent alternative for the specific reasons identified in Section 2. The interdependence of pricing decisions in supply chains is not a secondary consideration to be controlled away; it is the central feature of the problem. A research approach that models this interdependence explicitly is categorically more suitable than one that ignores it.

The specific choice of benchmarking environment is also well-judged. A fully synthetic environment would have made the results uninterpretable in practical terms. A live deployment on real customers would have introduced ethical and financial risks that no academic study can justify. The middle path, a simulation grounded in real historical data via a demand model, strikes the right balance between realism and experimental control.

Where the study falls short is in its transition from benchmark to insight. The comparative results are clearly presented, but the analysis stops short of asking why certain algorithms produce the behaviours they do. MADQN's aggression and low fairness, for instance, likely arise from its discrete action space and value-based update rule, but this is not examined. Connecting algorithmic mechanics to emergent market behaviour would elevate this study from a benchmarking exercise to a theoretical contribution.

Finally, the study's failure to address algorithmic collusion is a missed opportunity that borders on an ethical obligation. In a world where the EU AI Act explicitly addresses algorithmic pricing systems, and where regulators in multiple jurisdictions have opened investigations into automated pricing collusion, any serious study of MARL for competitive pricing must engage with this risk. That it does not is the study's most significant limitation.

VI.     Conclusions

Hazenberg et al. (2025) represents a meaningful advance in the application of reinforcement learning to dynamic pricing, not because it solves the problem, but because it asks for a more accurate version of it. By introducing multiple competing RL agents into a supply chain pricing environment grounded in real transactional data, the study reveals dynamics that single-agent formulations cannot capture: emergent strategic behaviour, competitive equilibria, and the spontaneous generation (or collapse) of market fairness without any normative design.

The central finding, that more adaptive agents produce less fair markets, is both empirically robust and theoretically important. It challenges the assumption that intelligence and social benefit are aligned, and raises practical questions about how competitive pricing systems should be regulated when they are driven by learning algorithms rather than fixed rules.

Read alongside the BBVA study of 2018, the two papers tell a coherent story about the field's evolution: from designing fairness into a reward function for a single agent, to understanding the conditions under which fairness does or does not emerge from multi-agent competition. Both perspectives are necessary. The BBVA approach tells us how to guarantee fair pricing when we control the system. The 2025 approach tells us what happens to fairness when multiple autonomous systems compete, which is the question that matters most for the real economy.

The open questions raised by this study, temporal resolution, collusion risk, domain generalizability, and continuous adaptation, are not weaknesses that invalidate the contribution. They are the agenda for the next generation of research in this area, which is precisely what an instructive paper should produce.


## References

[1] Hazenberg, T., Ma, Y., Mohammadi Ziabari, S.S., and van Rijswijk, M. (2025)...
[2] Maestre, R., Duque, J., Rubio, A., and Arevalo, J. (2018)...
[3] Sutton, R.S. and Barto, A.G. (2018)...
[4] Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., and Mordatch, I. (2017)...
[5] Mnih, V., Kavukcuoglu, K., Silver, D. et al. (2015)...
[6] Ke, G., Meng, Q., Finley, T. et al. (2017)...
[7] Rashid, T., Samvelyan, M., de Witt, C.S., Farquhar, G., Foerster, J., and Whiteson, S. (2018)...
[8] Cachon, G. and Feldman, P. (2010)...
[9] Jain, R., Chiu, D., and Hawe, W. (1984)...
[10] Chen, D. (2012)...

