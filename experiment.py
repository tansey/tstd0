import sys
import csv
from gridworld import *
import qlearning
import tstd



if __name__ == "__main__":
    build_rewards()
    outfile = sys.argv[1]
    bandits = int(sys.argv[2])
    episodes = int(sys.argv[3])
    world = GridWorld(num_bandits = bandits)
    """
    # TESTING WITH DETERMINISTIC WORLD
    world.bandits[UP].first = 1
    world.bandits[UP].second = 0
    world.bandits[RIGHT].first = 0
    world.bandits[RIGHT].second = 1
    world.bandits[DOWN].first = 0
    world.bandits[DOWN].second = 0
    """
    agents = [tstd.TSTDAgent(bandits), qlearning.QAgent(bandits)]
    series = ['Episodes', 'TSTD(0)', 'Q-Learning']
    row = [0 for _ in range(len(agents)+1)]
    f = open(outfile, 'wb')
    writer = csv.writer(f)
    writer.writerow(series)
    for ep in range(episodes):
        if ep % 10 == 0:
            print ep
        row[0] = ep + 1
        for i,agent in enumerate(agents):
            world.agent = agent
            score = world.play_episode()
            if i == 1:
                prev_epsilon = agent.epsilon
                agent.epsilon = 0
                score = world.play_episode()
                agent.epsilon = prev_epsilon
            row[i+1] = score
        writer.writerow(row)
    f.flush()
    f.close()
    print_state_values(agents[1].visits)
    print_q_values(agents[1].q)
    print_q_values(agents[1].q_visits)
