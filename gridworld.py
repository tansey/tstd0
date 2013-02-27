"""
Optimal action selection in the Multi-Armed Bandit Grid World.

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

MAB Grid World
- -4 - -4 - -4 --------
|  ^ |  ^ |  ^-1|     |
|    |    |   ->|     |
-----------------------
|    |    |   -1| +10 |
|    |    |   ->|     |
-----------------------
|    |    |   +3|     |
|    |    |   ->|     |
-----------------------

K = 10
Specific bandit distributions are randomly created at the start of every
experiment. The same set of distributions is used for both TSTD and Q-Learning
experiments.

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

Created by Wesley Tansey
2/24/2013
Code released under the MIT license.
"""
import random

# Actions an agent can take
UP = 0
RIGHT = 1
DOWN = 2

ACTION_NAMES = ['Up', 'Right', 'Down']

GRID_WIDTH = 4
GRID_HEIGHT = 3
START = (0, int(GRID_HEIGHT / 2))
GOAL = (GRID_WIDTH - 1, int(GRID_HEIGHT / 2))
GOAL_REWARD = 10
WALL_PENALTY = -4

REWARDS = { ((0,0),UP): ((0,0),-4) }

def build_rewards():
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            REWARDS[((x,y),UP)] = up_transition_reward(x,y)
            REWARDS[((x,y),RIGHT)] = right_transition_reward(x,y)
            REWARDS[((x,y),DOWN)] = down_transition_reward(x,y)

def up_transition_reward(x, y):
    if y == 0:
        return ((x, y),WALL_PENALTY)
    if x == GOAL[0] and y == GOAL[1]+1:
        return (GOAL,GOAL_REWARD)
    return ((x,y-1),0)

def right_transition_reward(x, y):
    if x == GRID_WIDTH - 1:
        return ((x,y),WALL_PENALTY)
    if x == GOAL[0]-1 and y == GOAL[1]:
        return (GOAL,GOAL_REWARD - 1)
    # Handle the special transition rewards
    if x == GRID_WIDTH - 2:
        if y == 0:
            return ((x+1,y), -1)
        else:
            return ((x+1,y), 3)
    return ((x+1,y),0)

def down_transition_reward(x, y):
    if y == GRID_HEIGHT - 1:
        return ((x, y), WALL_PENALTY)
    if x == GOAL[0] and y == GOAL[1]-1:
        return (GOAL, GOAL_REWARD)
    return ((x,y+1),0)



def print_world():
    print "-" * (12*GRID_WIDTH+1)
    for y in range(GRID_HEIGHT):
        if y != GOAL[1]:
            top = "|" + ("     ^     |"*GRID_WIDTH)
            bottom = "|" + ("     v     |"*GRID_WIDTH)
        else:
            top = "|" + ("     ^     |"*(GRID_WIDTH-1)) + "           |"
            bottom = "|" + ("     v     |"*(GRID_WIDTH-1)) + "           |"
        blank = "|" + ("           |"*GRID_WIDTH)
        up = "|"
        right = "|"
        down = "|"
        for x in range(GRID_WIDTH):
            if (x,y) == GOAL:
                up += "           |"
                right += "    GOAL   |"
                down += "           |"
            else:
                up += str(REWARDS[((x,y),UP)][1]).center(11) + "|"
                right += str(REWARDS[((x,y),RIGHT)][1]).rjust(9) + " >|"
                down += str(REWARDS[((x,y),DOWN)][1]).center(11) + "|"
        print top
        print up
        print blank
        print right
        print blank
        print down
        print bottom
        print "-" * (12*GRID_WIDTH+1)

def print_q_values(q):
    print "-" * (12*GRID_WIDTH+1)
    for y in range(GRID_HEIGHT):
        if y != GOAL[1]:
            top = "|" + ("     ^     |"*GRID_WIDTH)
            bottom = "|" + ("     v     |"*GRID_WIDTH)
        else:
            top = "|" + ("     ^     |"*(GRID_WIDTH-1)) + "           |"
            bottom = "|" + ("     v     |"*(GRID_WIDTH-1)) + "           |"
        blank = "|" + ("           |"*GRID_WIDTH)
        up = "|"
        right = "|"
        down = "|"
        for x in range(GRID_WIDTH):
            if (x,y) == GOAL:
                up += "           |"
                right += "    GOAL   |"
                down += "           |"
            else:
                up += "{0:0.2f}".format(q[((x,y),UP)]).center(11) + "|"
                right += "{0:0.2f}".format(q[((x,y),RIGHT)]).rjust(9) + " >|"
                down += "{0:0.2f}".format(q[((x,y),DOWN)]).center(11) + "|"
        print top
        print up
        print blank
        print right
        print blank
        print down
        print bottom
        print "-" * (12*GRID_WIDTH+1)

def print_state_values(v):
    print "-" * (12*GRID_WIDTH+1)
    for y in range(GRID_HEIGHT):
        blank = "|" + ("           |"*GRID_WIDTH)
        value = "|"
        for x in range(GRID_WIDTH):
            value += "{0:0.2f}".format(v[(x,y)]).center(11) + "|"
        print blank
        print blank
        print blank
        print value
        print blank
        print blank
        print blank
        print "-" * (12*GRID_WIDTH+1)

class Bandit(object):
    """
    A simple multinomial bandit. It returns one of three grid world actions:
    UP, RIGHT, or DOWN at every call of sample. Each sample of a specific
    bandit instance is IID.
    """
    def __init__(self):
        partition1 = random.random()
        partition2 = random.random()
        self.first = min(partition1, partition2)
        self.second = max(partition1, partition2) 

    def sample(self):
        r = random.random()
        if r < self.first:
            return UP
        if r < self.second:
            return RIGHT
        return DOWN

class Agent(object):
    """
    An abstract base class that grid world agents must extend.

    The agent is told how many bandits there are to choose from and what the
    reward function looks like for each state transition at the start of an
    episode. At every iteration of an episode, the agent is told its current
    (x,y) location in the world and it must return a bandit to sample an action
    from. After every action, the agent receives a reward from the environment.
    """
    def __init__(self, num_bandits):
        self.num_bandits = num_bandits

    def episode_starting(self, state):
        pass

    def episode_over(self):
        pass

    def get_bandit(self):
        pass

    def set_state(self, state):
        self.state = state

    def observe_action(self, action):
        pass

    def observe_reward(self, r):
        pass

class GridWorld(object):
    def __init__(self, max_moves = 100, num_bandits = 20, agent = None):
        self.max_moves = max_moves
        self.num_bandits = num_bandits
        self.agent = agent
        self.bandits = [Bandit() for _ in range(self.num_bandits)]

    def play_episode(self):
        state = START
        total_reward = 0
        self.agent.episode_starting(state)
        for i in range(self.max_moves):
            bidx = self.agent.get_bandit()
            assert(bidx >= 0)
            assert(bidx < self.num_bandits)
            bandit = self.bandits[bidx]
            action = bandit.sample()
            self.agent.observe_action(action)
            sr = REWARDS[(state, action)]
            total_reward += sr[1]
            self.agent.observe_reward(sr[1])
            state = sr[0]
            self.agent.set_state(state)
            if state == GOAL:
                break
        self.agent.episode_over()
        return total_reward

if __name__ == "__main__":
    build_rewards()
    print_world()
