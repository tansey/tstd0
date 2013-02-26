import csv
import os
import matplotlib.pyplot as plt
import sys

bandits = [2, 3, 5, 10, 20, 50, 100]
headers = ['Episodes','TSTD(0)', 'Q-Learning']
colors = ['blue','red','yellow', 'green', 'orange', 'purple', 'brown'] # max 7 lines
fileformat = "{0}_bandits_average.csv"

data = [[] for h in range(1,len(headers))]
for bandit in bandits:
    filename = fileformat.format(bandit)
    f = open(filename, 'r')
    reader = csv.reader(f)
    last = reader.next()
    for row in reader:
        last = row
    for i,v in enumerate(last):
        if i > 0:
            data[i-1].append(float(row[i]))

ax = plt.subplot(111)
for sidx,series in enumerate(data):
    plt.plot(bandits, series, label=headers[sidx+1], color=colors[sidx])
plt.xlabel("# Bandits")
plt.ylabel('Final Score')
plt.title(sys.argv[1])
# Shink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('summary.png')
plt.clf()
