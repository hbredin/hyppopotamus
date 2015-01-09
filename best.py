"""
Hyper-parameter tuning

Usage:
  best <mongo_host> <experiment.py>
  best -h | --help
  best --version

Options:
  -h --help            Show this screen.
  --version            Show version.
"""

from __future__ import print_function
from docopt import docopt

arguments = docopt(__doc__, version='')

# --- load experiment objective function and search space ---------------------
import sys
experiment_py = arguments['<experiment.py>']
execfile(experiment_py)

# --- load experiment trials --------------------------------------------------
host = arguments['<mongo_host>']

from hyperopt.mongoexp import MongoTrials
trials = MongoTrials(
    'mongo://{host}/{xp_name}/jobs'.format(host=host, xp_name=xp_name))

# --- show results ------------------------------------------------------------
from hyperopt import space_eval
import numpy as np
from pprint import pprint

print('#> BEST LOSS')
best = trials.best_trial
result = dict(best['result'])
result.pop('status')
pprint(result)

print('#> BEST PARAMETERS')
pprint(space_eval(xp_space, trials.argmin))
