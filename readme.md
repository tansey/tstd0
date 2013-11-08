# Thompson-Sampling + TD(0)

Optimal action selection in the Multi-Armed Bandit Grid World.

## The Multi-Armed Bandit (MAB) Grid World

In the MAB Grid World, the agent's goal is to learn an optimal policy for
navigating an episodic grid. The twist here is that the agent has full
knowledge of the reward and transition functions, but its actions are 
indirectly chosen.

There are K bandits, {X_1, X_2, ... X_k} from which the agent can choose at
every step. The bandits remain the same no matter what state the agent is in.
Each bandit will then return a randomly sampled action from an unknown
distribution. The agent must learn a policy pi that chooses the optimal bandit
at every step.

Specifically to this world, there are three categories of actions: up,
down, and right. The agent starts at the left-most center square and the
goal state is the right-most center square.

See `gridworld.py` for an ASCII depiction of the grid world.

Specific bandit distributions are randomly created at the start of every
experiment. The same set of distributions is used for both TSTD and Q-Learning
experiments.

## TSTD

The approach we take here is to use a generalized Thompson Sampling approach
combined with a simple TD(0) method. We keep the TD method as simple as
possible so as to make a fair comparison with out baseline method: Q-learning.
Note that standard Thompson Sampling assumes univariate values in the range
[0,1]. In this case, we want to predict values from a categorical distribution
and thus rather than using a Beta distribution to model our prior and posterior,
we use a Dirichlet distribution.

The advantage of Thompson Sampling TD (TSTD) is the huge reduction in state
action pairs. Methods like Q-learning will model the full (s,a) pairs and
not take the model into account.

## Attribution

Created by Wesley Tansey

Code released under the MIT license.