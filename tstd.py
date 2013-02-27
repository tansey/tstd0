from itertools import product
import random
from gridworld import *

class TSTDAgent(Agent):
    """
    An agent that uses Thompson Sampling with TD(0)
    to minimize regret.

    TODO: How should Q-Learning be incorporated here?
    """
    def __init__(self, num_bandits, alpha = 1, decrease_alpha = True):
        Agent.__init__(self, num_bandits)
        # Three values: UP, RIGHT, DOWN. We use a uniform prior.
        self.priors = [[1,1,1] for x in range(num_bandits)]
        self.build_value_table()
        self.episodes = 0
        self.alpha = alpha
        self.decrease_alpha = decrease_alpha
        self.starting_alpha = alpha

    def sample_dirichlet(self, prior):
        sample = [random.gammavariate(a,1) for a in prior]
        sample = [v/sum(sample) for v in sample]
        return sample

    def build_value_table(self):
        self.v = {}
        for state in product(range(GRID_WIDTH), range(GRID_HEIGHT)):
            self.v[state] = 0

    def episode_starting(self, state):
        self.state = state
        self.prev_state = None
        
    def episode_over(self):
        self.episodes += 1
        if self.decrease_alpha:
            self.alpha = min(self.starting_alpha, 20.0 / float(self.episodes+1))

    def get_bandit(self):
        return self.thompson_sampling()[0]

    def thompson_sampling(self):
        samples = [self.sample_dirichlet(prior) for prior in self.priors]
        regret = [self.calc_regret(sample) for sample in samples]
        maxi = 0
        for i in range(self.num_bandits):
            if regret[i] < regret[maxi]:
                maxi = i
        self.prev_bandit = maxi
        return (maxi,samples[maxi])

    def calc_regret(self, bandit):
        """
        Calculates the regret of choosing this bandit
        """
        srup = REWARDS[(self.state,UP)]
        srright = REWARDS[(self.state,RIGHT)]
        srdown = REWARDS[(self.state,DOWN)]
        qup = srup[1] + self.v[srup[0]]
        qright = srright[1] + self.v[srright[0]]
        qdown = srdown[1] + self.v[srdown[0]]
        qmax = max(qup, qright, qdown)
        regret = bandit[UP] * (qmax - qup)
        regret += bandit[RIGHT] * (qmax - qright)
        regret += bandit[DOWN] * (qmax - qdown)
        return regret

    def set_state(self, state):
        # First we need to update the value for the previous state.
        self.update_v()
        self.prev_state = self.state
        self.state = state

    def update_v(self):
        # We've updated our priors on bandits, so we need to know what our new min-regret bandit is
        bandit = self.thompson_sampling()[1]
        srup = REWARDS[(self.state,UP)]
        srright = REWARDS[(self.state,RIGHT)]
        srdown = REWARDS[(self.state,DOWN)]
        val =  bandit[UP]*(self.v[srup[0]]+srup[1])
        val += bandit[RIGHT]*(self.v[srright[0]]+srright[1])
        val += bandit[DOWN]*(self.v[srdown[0]]+srdown[1])
        self.v[self.state] = (1.0-self.alpha) * self.v[self.state] + self.alpha * val

    def observe_action(self, action):
        self.priors[self.prev_bandit][action] += 1
        self.prev_action = action

    def observe_reward(self, r):
        self.prev_reward = r



