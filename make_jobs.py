import sys
import os

def make_directory(base, subdir):
    if not base.endswith('/'):
        base += '/'
    directory = base + subdir
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

if len(sys.argv) < 4 or not sys.argv[1].endswith('.py'):
    print 'Format: python makejobs.py <main_file.py> <experiment_name> <trials> <experiment description> [main_file.py parameters]'
    print 'Note: The first parameter that main_file.py takes MUST be a csv file name where it will write the results'
    exit(1)

main_file = sys.argv[1]
experiment_name = sys.argv[2]
trials = int(sys.argv[3])
experiment_dir = make_directory(os.getcwd(), experiment_name.replace(' ', '_'))
jobsfile = experiment_dir + '/jobs'
desc = sys.argv[4]

if len(sys.argv) > 5:
    main_file_params = " ".join([str(x) for x in sys.argv[5:]])
else:
    main_file_params = ""

make_directory(experiment_dir, 'condor_logs')
make_directory(experiment_dir, 'results')
make_directory(experiment_dir, 'output')
make_directory(experiment_dir, 'error')

f = open(jobsfile, 'wb')
f.write("""universe = vanilla
Executable=/lusr/bin/python
+Group   = "GRAD"
+Project = "AI_ROBOTICS"
+ProjectDescription = "{0}"
""".format(desc))

job = """Log = {0}/condor_logs/job_{2}.log
Arguments = {3} {0}/results/{1}_{2}.csv {4}
Output = {0}/output/output_{2}.out
Error = {0}/error/error_{2}.log
Queue 1
"""

for trial in range(trials):
    f.write(job.format(experiment_dir, experiment_name.replace(' ', '_'),  trial, main_file, main_file_params))
    
f.flush()
f.close()
