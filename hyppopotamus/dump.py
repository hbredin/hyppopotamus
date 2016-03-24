"""
Hyper-parameter tuning

Usage:
  dump <mongo_host> <experiment.py> <output.dir>
  dump -h | --help
  dump --version

Options:
  -h --help            Show this screen.
  --version            Show version.
"""

from __future__ import print_function
from docopt import docopt

arguments = docopt(__doc__, version='')

# --- load experiment objective function and search space ---------------------
experiment_py = arguments['<experiment.py>']
execfile(experiment_py)

# --- load experiment trials --------------------------------------------------
host = arguments['<mongo_host>']

from hyperopt.mongoexp import MongoTrials
trials = MongoTrials(
    'mongo://{host}/{xp_name}/jobs'.format(host=host, xp_name=xp_name))

# --- dump best trial attachments ---------------------------------------------
output_dir = arguments['<output_dir>']

xp_dump(
    dict(trials.trial_attachments(trial=trials.best_trial)),
    output_dir)
