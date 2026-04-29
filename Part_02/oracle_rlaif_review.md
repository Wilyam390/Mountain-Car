REINFORCEMENT LEARNING INTRODUCTION
Assignment 22.00 - Group Assignment | Part 02

# Analytical review

## Oracle-RLAIF: Reinforcement Learning from Oracle Ranking Feedback

---

## Abstract

This paper examines Oracle-RLAIF, a novel reinforcement learning framework introduced by Shi et al. (2025) from Lawrence Livermore National Laboratory, designed to fine-tune large video-language models (VLMs) using AI-generated ranking feedback rather than human annotations or score-based reward models. The paper addresses a critical bottleneck in modern multi-modal AI alignment: the prohibitive cost and inflexibility of training calibrated reward models for reinforcement learning from human feedback (RLHF) and existing reinforcement learning from AI feedback (RLAIF) approaches. The proposed framework replaces the trained reward model with a drop-in Oracle ranker that only needs to order candidate responses by quality rather than assign scalar scores. Alongside this, the authors introduce GRPOrank, a novel rank-adapted version of Group Relative Policy Optimization (GRPO) that directly incorporates ordinal feedback into the policy gradient objective using a Normalized Discounted Cumulative Gain penalty. Empirical results across MSVD, MSRVTT, ActivityNet, and Video-MME benchmarks demonstrate consistent improvements over the current VLM-RLAIF framework, with a +6.2% average accuracy gain on Video-MME and particularly strong improvements in temporal perception (+21.2%) and action recognition (+11.7%). This review situates the paper within the broader RL literature, explains all core RL concepts employed, and critically examines the methodology, experimental design, and implications of the findings.

---

## Introduction

The alignment of large language and vision-language models with human intent has become one of the central challenges in modern artificial intelligence. State-of-the-art multi-modal foundation models, systems capable of processing both visual (video/image) and textual inputs, are increasingly trained using a two-phase pipeline: supervised fine-tuning followed by a reinforcement learning phase that incorporates human or AI preference signals. This two-stage approach has proven essential because while SFT ensures syntactically coherent and topically relevant outputs, it does not guarantee that the model's responses truly align with what a human would judge as the best or most informative answer. The reinforcement learning phase fills this gap by using preference data to guide the policy model toward higher-quality outputs.

Reinforcement Learning from Human Feedback (RLHF); most famously applied in InstructGPT and ChatGPT, formalizes this by training a reward model on human pairwise comparisons of model outputs, then optimizing the language model policy using Proximal Policy Optimization (PPO) to maximize expected reward. However, as video-language models scale to billions of parameters and are applied to increasingly complex video understanding tasks, gathering sufficient high-quality human feedback becomes economically and logistically prohibitive.

Reinforcement Learning from AI Feedback (RLAIF) emerged as a scalable alternative, replacing human annotators with a capable AI judge. The VLM-RLAIF framework (Ahn et al., 2024) applied this to video-language models, demonstrating significant gains in video question answering. However, their approach still requires training a large, specialized 13-billion-parameter reward model using video narrative data harvested from over 99,000 ActivityNet videos; a pipeline that is expensive, dataset-specific, and tightly coupled to a particular scoring mechanism.

Oracle-RLAIF tackles these limitations by decoupling the fine-tuning framework from the requirement of a calibrated scalar reward model. By replacing the trained reward model with a general Oracle ranker,  any model capable of ordering responses by quality, the authors create a more flexible, data-efficient, and broadly applicable alignment pipeline. The theoretical innovation is matched by a practical algorithmic contribution: GRPOrank, which extends GRPO to handle ordinal rank signals directly in the policy optimization objective, without converting rankings to scores.

This literature review systematically unpacks every RL concept, design choice, and empirical finding presented in the paper, providing a comprehensive analysis.

---

## Problem Identification and Solution Suitability

### The Core Problem

The paper identifies two tightly coupled problems:

Problem 1: Reward Model Brittleness: Existing RLAIF frameworks depend on a reward model trained with rich contextual data (video narratives, scored preference pairs). Building such a model requires extensive domain-specific data collection. Moreover, score-based reward models can be inconsistent, a model may score "Response A: 0.82" and "Response B: 0.45" in a way that is inconsistent across different queries, making reinforcement learning unstable (the "accuracy paradox" noted by Chen et al., 2024).

Problem 2: Inflexibility of Score-Based RL: Scalar reward signals must be calibrated (their magnitude must be meaningful). This is inherently restrictive: it prevents the use of general-purpose closed-source models (like GPT-4) as judges, since they can rank responses but their scores are not reliably calibrated for RL training.

### Why Ranking is a Better Signal

Ranking is a weaker but more robust form of supervision: Any model,  including general closed-source LLMs or legacy systems, can rank responses without needing calibrated scores. Relative ordering is easier to elicit reliably than absolute scoring. The ordinal structure of rankings naturally encodes which responses to promote and which to suppress, without requiring the magnitude of rewards to be consistent.

### Oracle-RLAIF's Proposed Solution

The authors' key insight is that ranking is strictly easier than scoring. A ranker only needs to answer 'which response is better?' rather than 'how good is this response on a scale of 0 to 5?' This relaxation makes the Oracle far easier to implement as a drop-in component, any capable general-purpose language model can serve as the Oracle, including closed-source commercial models. This also eliminates the need for specialized reward model training and the associated data collection pipeline.

The solution is thus two-fold: (1) replace the trained reward model with a drop-in Oracle ranker in the RLAIF loop, and (2) develop a policy optimization algorithm (GRPOrank) that can directly consume rank signals rather than requiring their conversion to scalar rewards.

---

## Methodology

### Pipeline Overview


The Oracle-RLAIF pipeline is a three-stage iterative loop. It begins with a VLM already fine-tuned via supervised learning (SFT) on video–question–answer data, which serves as the initial policy π_θ. At each training epoch, the current policy generates G = 5 candidate responses per prompt; the Oracle ranker orders them from best (rank 0) to worst; and GRPOrank updates the policy to increase the probability of higher-ranked responses. Because responses are always drawn from the current policy, not a replay buffer, the loop is fully on-policy, ensuring feedback stays calibrated to the model's present behaviour.
Oracle Ranker
The Oracle is trained on the same preference pairs as the VLM-RLAIF reward model, but deliberately omits video narrative caption data (99,000+ ActivityNet captions used by the baseline). This simulates a drop-in, general-purpose ranker requiring no domain-specific fine-tuning. The key relaxation is that the Oracle only needs to order responses, not assign calibrated scalar scores; a strictly easier task that demands less data and less calibration effort, and that allows any capable language model to serve as the Oracle, including closed-source commercial models.
 Training Infrastructure
Both models start from the same VLM-SFT 7B checkpoint (LLaMA-2-7B + frozen CLIP ViT-L/14-336, connected via a 32-token Q-Former adapter). Efficient fine-tuning is achieved through QLoRA, 4-bit quantized base weights with low-rank adapters (rank 65, α = 16),  enabling training on 4× NVIDIA H100 80 GB GPUs within a single node. Training runs for 4 epochs with a rollout batch size of 64.

### Data Foundation and Demand Modelling
The SFT base model was trained on ~327,000 video–text samples: ~80,000 synthetically generated instruction-tuning samples (Video-ChatGPT, PandaGPT), ~67,000 video QA samples (NExT-QA, LLaVA), and ~180,000 object-centric samples for visual grounding. A curriculum learning strategy organises these by difficulty, measured by answer length, progressing from 214k easier samples to 113k harder multi-hop comprehension tasks. For the RL phase, preference pairs from VLM-SFT outputs are used as prompts, with the Oracle assigning ranks across 5 generated candidates per query.
Demand refers to the distribution of video understanding capabilities the model must master. The paper identifies six dimensions across its benchmarks: action recognition, temporal reasoning, object recognition and reasoning, causal understanding, spatial perception and reasoning, and information synthesis. This decomposition is analytically important: Oracle-RLAIF's rank-based signal proves unequally effective across these dimensions, with strong gains in temporally grounded tasks (+21.2% Temporal Perception, +11.7% Action Recognition on Video-MME) but performance drops in spatial and abstract reasoning, a limitation the authors attribute to higher ambiguity in those categories, where rank differences between candidates are less consistent.

### Simulation Environment Design
Oracle-RLAIF does not use a classical simulator. The environment is the RL fine-tuning loop itself, operating over the video understanding task space. Each episode is a single (video, prompt), response generation sequence. The state at step t is the video frames plus all tokens generated so far; the action is the next token chosen from the vocabulary; transitions are deterministic (the new state is simply the prior context extended by the chosen token); and the reward is sparse and delayed, realized only once the full response is complete and ranked by the Oracle. Two evaluation environments are used: open-ended QA on MSVD, MSRVTT, and ActivityNet (scored by GPT-3.5-turbo on five dimensions), and multiple-choice QA on Video-MME; the more rigorous setting, as it uses objective accuracy and was explicitly designed to avoid training data leakage.

### MDP Formulation

The fine-tuning process is formally a Markov Decision Process over token generation sequences. The table below summarises its components.

| Component     | Definition                                                                                                                                                       |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| State  S      | s_t = (video frames, prompt, tokens generated so far o_{i,<t}). Fully observable to the policy.                                                                  |
| Action  A     | a_t = next token ∈ Vocabulary (~32k–128k tokens). A full response is the sequence of actions until end-of-sequence.                                              |
| Transition  T | Deterministic: s_{t+1} appends the chosen token to the current context. No stochastic dynamics.                                                                  |
| Reward  R     | Sparse and delayed, realised only at episode end. The Oracle assigns rank r_i ∈ {0,…,K−1} (0 = best); this drives the nDCG-based advantage. No per-token reward. |
| Discount  γ   | γ = 1 (undiscounted). The rank-derived advantage is applied uniformly across all tokens in the response, standard for GRPO-family LLM fine-tuning.               |
| Policy  π_θ   | π_θ(o_{i,t} | video, prompt, o_{i,<t}), the VLM's token distribution, parameterised by trainable QLoRA adapter weights θ.                                        |

The objective is to find θ* maximising expected Oracle-approved response quality across the prompt distribution, subject to a KL divergence bound that prevents excessive drift from the reference policy frozen at the start of each epoch.



### Algorithm Selection and Architecture

PPO, the algorithm used by VLM-RLAIF, requires two components unavailable here: a learned value function V_φ(s_t) to estimate baseline-adjusted advantages, and scalar reward scores to train it. Oracle-RLAIF has neither; it produces only ordinal ranks. GRPO (Shao et al., 2024) removes the value function by sampling G responses per query and computing advantages relative to the group mean, eliminating one dependency. However, standard GRPO still expects scalar rewards R(q, o_i). GRPOrank eliminates the second dependency by replacing scalar rewards with a rank-derived advantage grounded in Normalized Discounted Cumulative Gain (nDCG).

For each query, the Oracle assigns ranks to G = 5 candidate responses. The policy's own log-probabilities implicitly define a predicted ranking (higher probability = predicted better). The nDCG penalty for response i measures the gap between these two rankings:

δ_i = 1 − DCG(r̂_i) / DCG(rank_i)     where  DCG(rank) = 1 / ((1 + rank) · log₂(2 + rank))

The rank-adapted advantage is then the response's penalty relative to the group mean:

Â_rank = E_{j∈G}[δ_j] − δ_i

This replaces the normalized scalar reward in GRPO. The full GRPOrank loss adds KL regularization (to bound drift from the reference policy) and an entropy bonus (to prevent premature collapse):

L_GRPOrank = clipped surrogate(Â_rank) − β · D_KL(π_θ_old ‖ π_θ) + c_ent · H[π_θ(·|q)]



### Why GRPOrank Works: Three Properties?

Zero-sum within groups. Advantages sum to zero across all G responses (Σ Â_rank = 0), so optimisation is purely comparative. This prevents reward drift and mode collapse.

Bounded penalties. Since nDCG ∈ (0,1], penalties δ_i ∈ [0,1); numerically stable gradients with no outlier blowup.

Position-sensitive discounting. The logarithmic DCG formula penalises top-rank errors far more heavily than bottom-rank errors: misidentifying the best response as second-best (δ = 0.21) costs 9× more than confusing fourth and fifth place (δ = 0.02), and misranking the best response last yields the maximum penalty (δ = 0.48). This directly encodes the practical priority that correctly identifying the best response matters most.

---

Together, these properties make GRPOrank a principled drop-in replacement for scalar-reward GRPO, preserving the group-relative advantage structure while extending it to ordinal feedback, and doing so without any additional model components or scalar reward calibration.

---

## Results and Interpretation

### Benchmark Performance

| Benchmark             | Oracle-RLAIF | VLM-RLAIF | Δ      |
| --------------------- | ------------ | --------- | ------ |
| MSVD-QA (Acc.)        | 72.9%        | 68.5%     | +4.4%  |
| MSRVTT-QA (Acc.)      | 59.2%        | 54.2%     | +5.0%  |
| ActivityNet-QA (Acc.) | 48.1%        | 46.1%     | +2.0%  |
| Video-MME Overall     | 42.4%        | 36.2%     | +6.2%  |
| Temporal Perception   | 71.0%        | 49.8%     | +21.2% |
| Action Recognition    | 39.0%        | 27.3%     | +11.7% |
| Object Reasoning      | 49.5%        | 38.3%     | +11.2% |

The results confirm that Oracle-RLAIF's rank-based optimization is superior for temporal and action-based video understanding, representing a significant advancement in the RLAIF paradigm.

### Limitations and Failure Modes
Oracle-RLAIF showed performance decreases relative to VLM-RLAIF on three Video-MME categories: Spatial Perception (−2.6%), Spatial Reasoning (−3.8%), and Information Synopsis (−0.9%). The authors hypothesize that these categories involve higher ambiguity and implicit reasoning that rank-based optimization cannot effectively capture; for instance, inferring a video's holiday setting from ambient lighting and background decor requires holistic contextual understanding that may not be well-separated across candidate responses by rank alone. This suggests that GRPOrank's strength lies in scenarios where quality differences between responses are clear and consistent,  exactly the case for action recognition and temporal reasoning tasks.
Broader RL Takeaways
From a reinforcement learning perspective, Oracle-RLAIF makes several important contributions to the RL-for-LLM alignment literature:
It demonstrates that ordinal feedback is sufficient for effective policy optimization, challenging the assumption that scalar rewards are necessary for RL-based alignment.
The use of nDCG as a penalty function elegantly encodes position-sensitive preferences into the advantage function, drawing a productive connection between information retrieval theory and RL.
The zero-sum group advantage structure prevents the degenerate solutions that scalar GRPO can produce when rewards are poorly calibrated or biased.
The drop-in Oracle design significantly lowers the barrier to applying RL-based alignment, potentially enabling fine-tuning with closed-source commercial models as the Oracle without any domain-specific reward model training.

---

## Critical Personal Evaluation

Oracle-RLAIF is a clever piece of engineering, but the paper oversells its novelty. Ranking being easier than scoring is not a new insight; it is foundational to learning-to-rank literature. The real contribution is operationalizing that insight inside a GRPO-style RL loop via nDCG penalties, which is genuinely useful but more incremental than the framing suggests.
The most significant flaw the paper sidesteps is Oracle quality. The entire framework rests on the Oracle producing reliable rankings, yet there is no ablation testing what happens with a weaker or noisier Oracle, precisely the robustness test that matters most for a framework whose central claim is Oracle flexibility.
The spatial reasoning failures are also more theoretically damaging than treated. A −3.8% drop on Spatial Reasoning is not a minor limitation; it reveals a structural assumption baked into GRPOrank, that better responses are consistently rankable, which breaks down for holistically ambiguous tasks. The paper identifies the symptom but proposes no mechanism to detect or handle it.
That said, +21.2% on Temporal Perception on an objective benchmark with no training overlap is hard to dismiss. The alignment between nDCG's position weighting and the natural priority of video QA, identifying the best response matters more than ordering mediocre ones, feels principled rather than accidental.
The contribution is real. The framing is slightly inflated.


---

## Conclusions

Oracle-RLAIF represents a meaningful advance in the practical application of reinforcement learning to multi-modal model alignment. By relaxing the reward modeling requirement from scalar scoring to ordinal ranking, and by developing the GRPOrank algorithm to exploit that signal directly in policy optimization, the authors produce a more flexible and data-efficient alignment framework that outperforms the current state-of-the-art on most video understanding benchmarks.
The work is grounded in solid RL theory, the MDP formulation is clear, the policy gradient derivation is principled, and the mathematical properties of GRPOrank are rigorously established. The empirical evaluation is thorough, with a meaningful second experiment on Video-MME that avoids the data leakage concerns present in the first. The identified failure modes on spatial and abstract reasoning tasks point to productive directions for future work.
From the perspective of an RL course, this paper is an excellent example of how core RL concepts, MDPs, policy gradients, advantage estimation, on-policy learning, KL regularization, are adapted and extended to address real-world alignment challenges in large-scale generative models. It illustrates that thoughtful choice of feedback representation (ranks vs. scores) can have profound implications for algorithm design, computational cost, and downstream performance.

---

## References

Main focus:
Shi, D., Glatt, R., Klymko, C., et al. (2025)...

Supporting Context:

Ahn, D., Hu, Y., Ostapenko, O., et al. (2024). Tuning large multimodal models for videos using reinforcement learning from AI feedback. ACL 2024.

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. NeurIPS 2023.

DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv:2501.12948.

Fu, C., Dai, Y., Luo, Y., et al. (2025). Video-MME: The first-ever comprehensive evaluation benchmark of multi-modal LLMs in video analysis. CVPR 2025.

Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training language models to follow instructions with human feedback. NeurIPS 2022.

Schulman, J., Wolski, F., Dhariwal, P., et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.

Shao, Z., Wang, P., Zhu, Q., et al. (2024). DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv:2402.03300.
