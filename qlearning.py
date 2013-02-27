import random
from itertools import product
from gridworld import *

class QAgent(Agent):
    """
    A Q-Learning agent with optimistic initialization.
    """
    def __init__(self, num_bandits, epsilon = 0.1, decrease_epsilon = False, alpha = 0.05, decrease_alpha = False, gamma = 1):
        Agent.__init__(self, num_bandits)
        self.build_state_action_table()
        self.epsilon = epsilon
        self.alpha = alpha
        self.decrease_epsilon = decrease_epsilon
        self.decrease_alpha = decrease_alpha
        self.gamma = gamma
        self.episodes = 0
        self.starting_epsilon = epsilon
        self.starting_alpha = alpha

    def build_state_action_table(self):
        self.q = {}
        self.visits = {}
        self.q_visits = {}
        for sa in product(product(range(GRID_WIDTH), range(GRID_HEIGHT)), range(self.num_bandits)):
            self.q[sa] = 0
            self.q_visits[sa] = 0
        for s in product(range(GRID_WIDTH), range(GRID_HEIGHT)):
            self.visits[s] = 0

    def episode_starting(self, state):
        self.state = state
        self.update = False
        
    def episode_over(self):
        if self.update:
            self.q[(self.prev_state,self.prev_action)] += self.alpha * (self.prev_reward - self.q[(self.prev_state,self.prev_action)])
        self.episodes += 1
        if self.decrease_alpha:
            self.alpha = min(self.starting_alpha, 20.0 / float(self.episodes+1))
        if self.decrease_epsilon:
            self.epsilon = min(self.starting_epsilon, 100.0 / float(self.episodes+1))

    def get_bandit(self):
        if self.update:
            self.update_q()
        if random.random() < self.epsilon:
            self.update = True
            bandit = random.randrange(0,self.num_bandits)
        else:
            self.update = True
            (bandit, bval) = self.greedy()
        self.prev_action = bandit
        self.q_visits[(self.state,bandit)] += 1
        return bandit

    def greedy(self,debug=False):
        maxi = []
        maxv = 0
        for i in range(self.num_bandits):
            bval = self.q[(self.state,i)]
            if debug:
                print '\tQ[s,{0}] = {1}'.format(ACTION_NAMES[i], bval)
            if len(maxi) == 0 or bval > maxv:
                maxi = [i]
                maxv = bval
            elif bval == maxv:
                maxi.append(i)
        if len(maxi) == 1:
            maxi = maxi[0]
        else:
            maxi = random.choice(maxi)
        if debug:
            print '\tChoosing: {0}'.format(ACTION_NAMES[maxi])
        return (maxi,maxv)

    def set_state(self, state):
        self.prev_state = self.state
        self.state = state
        self.visits[state] += 1

    def update_q(self):
        target = self.prev_reward + self.gamma * self.greedy()[1] - self.q[(self.prev_state,self.prev_action)]
        self.q[(self.prev_state,self.prev_action)] += self.alpha * (self.prev_reward + self.gamma * self.greedy()[1] - self.q[(self.prev_state,self.prev_action)])

    def observe_action(self, action):
        """
        Q-learning does not utilize the observed action taken by the bandits. It only optimizes directly.
        """
        self.prev_move = action

    def observe_reward(self, r):
        self.prev_reward = r
