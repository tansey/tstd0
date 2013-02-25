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
    world = GridWorld()
    agents = [qlearning.QAgent(bandits), tstd.TSTDAgent(bandits)]
    series = ['Episodes', 'Q-Learning', 'TSTD(0)']
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
            row[i+1] = world.play_episode()
        writer.writerow(row)
    f.flush()
    f.close()
    print '**** Q-Values for Q-Learning ****'
    print_q_values(agents[0].q)
    print '**** State values for TSTD ****'
    print_state_values(agents[1].v)