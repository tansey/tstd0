import csv
import os
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import sys

def initialize_results(filename):
    f = open(filename, 'r')
    reader = csv.reader(f)
    initial_results = []
    reader.next()
    for row in reader:
        initial_results.append([[] for _ in range(len(row))])
    f.flush()
    f.close()
    return initial_results

def get_headers(filename):
    tempf = open(filename, 'r')
    reader = csv.reader(tempf)
    headers = reader.next()
    tempf.close()
    return headers

def write_results(filename, r):
    f = open(filename, 'wb')
    writer = csv.writer(f)
    for row in r:
        writer.writerow(row)
    f.flush()
    f.close()
    
def aggregate_results():
    results_dir = sys.argv[1]
    if not results_dir.endswith('/'):
        results_dir += '/'
    setup = True
    trials = 0
    for filename in os.listdir(results_dir):
        if setup:
            tokens = filename.replace(".csv", "").split('_')
            experiment_name = ""
            for i in range(len(tokens) - 1):
                if i > 0:
                    experiment_name += " "
                experiment_name += tokens[i]
            results = initialize_results(results_dir + filename)
            headers = get_headers(results_dir + filename)
            setup = False
        filename = results_dir + filename
        f = open(filename, 'r')
        reader = csv.reader(f)
        reader.next()
        for i,row in enumerate(reader):
            for j,val in enumerate(row):
                results[i][j].append(float(val))
        f.flush()
        f.close()
        trials += 1

    avg = []
    stdev = []
    stderr = []
    for row in results:
        avg.append([])
        stdev.append([])
        stderr.append([])
        for cell in row:
            avg[-1].append(sum(cell) / len(cell))
            stdev[-1].append(sqrt(sum([(x - avg[-1][-1])**2 for x in cell]) / (len(cell) - 1)))
            stderr[-1].append(stdev[-1][-1] / sqrt(len(cell)))

    write_results(experiment_name.replace(' ', '_') + '_average.csv', avg)
    write_results(experiment_name.replace(' ', '_') + '_stdev.csv', stdev)
    write_results(experiment_name.replace(' ', '_') + '_stderr.csv', stderr)
    plot(sys.argv[1], experiment_name, avg, stdev, stderr, trials, headers)
    
def transpose_results(results):
    t = [[] for _ in range(len(results[0]))]
    for row in results:
        for i,cell in enumerate(row):
            t[i].append(cell)
    return t
    
def plot(graph_title, experiment_name, avg, stdev, stderr, trials, series):
    avg = [np.array(x) for x in transpose_results(avg)]
    stdev = [np.array(x) for x in transpose_results(stdev)]
    stderr = [np.array(x) for x in transpose_results(stderr)]
    colors = ['blue','red','yellow', 'green', 'orange', 'purple', 'brown'] # max 7 lines
    ax = plt.subplot(111)
    for i in range(1,len(avg)):
        plt.plot(avg[0], avg[i], label=series[i], color=colors[i-1])
        plt.fill_between(avg[0], avg[i] + stderr[i], avg[i] - stderr[i], facecolor=colors[i-1], alpha=0.2)
    plt.xlabel(series[0])
    # Note the code below is experiment-specific
    plt.ylabel('Score')
    plt.title('{0}\n({1}, {2} trials)'.format(graph_title, experiment_name, trials))
    # Shink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('{0}.png'.format(experiment_name.replace(' ', '_')))
        
if len(sys.argv) != 3:
    print "Format: python plot_results.py <results_dir> <graph_title>"
    exit(1)
print "Plotting {0} {1}".format(sys.argv[1], sys.argv[2])
aggregate_results()
        
        

