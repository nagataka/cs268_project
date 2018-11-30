# Project

PPO is motivated by the same question as TRPO: how can we take the biggest possible improvement step on a policy using the data we currently have, without stepping so far that we accidentally cause performance collapse? Where TRPO tries to solve this problem with a complex second-order method, PPO is a family of first-order methods that use a few other tricks to keep new policies close to old.

There are two primary variants of PPO: PPO-Penalty and PPO-Clip.

- **PPO-Penalty** approximately solves a KL-constrained update like TRPO, but penalizes the KL-divergence in the objective function instead of making it a hard constraint, and automatically adjusts the penalty coefficient over the course of training so that it’s scaled appropriately.

- **PPO-Clip** doesn’t have a KL-divergence term in the objective and doesn’t have a constraint at all. Instead relies on specialized clipping in the objective function to remove incentives for the new policy to get far from the old policy.

(^^^Quote from https://spinningup.openai.com/en/latest/algorithms/ppo.html#spinup.ppo)

#### Project objective(s)

In our project, we investigate these two variants and compare the performance, in particular, although they mentioned PPO-Penalty, they focued on PPO-Clip in their papre so the motivation here is to shed light on PPO-Penalty and dive deep into the theoretical comparison of them.

(Optional) In addition to this, we'll study how hyperparameters contribute to (or affect) the performance.

- Code PPO from scratch
- Performance comparison (env: Luna Lander)
    - $L^{CLIP}$ and $L^{KLPEN}$
    - Different num of workers and investigate how linearly scale the PPO is
    - Different hyperparameters
        - Repro the Table 1
- (Compare DDPG + HER vs. PPO + HER)
- Compare with DDPG + RND and PPO + RND
    - implement RND
        - Be sure to normalize reward

## Brief review on PPO

PPO is one of the policy gradient methods which is originaly proposed by OpenAI which uses only first-order optimization (wheres TRPO, their precedindg method, is a second-order method).

##### Policy Gradient Methods


##### Trust Region Methods

(Maybe good to quickly study trust region method in standard Optimization context to check if the term is used a bit differently)





## CLIP and KL



## Hyper parameters


##### Refs

- [Training with Proximal Policy Optimization](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-PPO.md)
- [Best Practices when training with PPO](https://github.com/EmbersArc/PPO/blob/master/best-practices-ppo.md)

#### 

### Reference code

- General
    - [Spinningup](https://spinningup.openai.com/en/latest/index.html)
- KL penalty
    - https://github.com/SSS135/pytorch-rl-kit
    - https://github.com/joschu/modular_rl
    - https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/12_Proximal_Policy_Optimization/simply_PPO.py


##### Which one ($L^{CLIP} or L^{KLPEN}$) the code implemented?
