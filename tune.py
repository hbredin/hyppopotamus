"""
Hyper-parameter tuning

Usage:
  tune [--max=<max_evals>] [--mongo=<host>] <experiment.py>
  tune -h | --help
  tune --version

Options:
  --mongo=<host>       Set MongoDB host.
  --max=<max_evals>    Allow up to this many evaluations [default: 100]
  -h --help            Show this screen.
  --version            Show version.
"""

from docopt import docopt

arguments = docopt(__doc__, version='')

# --- load experiment objective function and search space ---------------------
import sys
experiment_py = arguments['<experiment.py>']
execfile(experiment_py)

# --- load experiment trials --------------------------------------------------
host = arguments['--mongo']

if host is None:
    from hyperopt import Trials
    trials = Trials()

else:
    from hyperopt.mongoexp import MongoTrials
    trials = MongoTrials(
        'mongo://{host}/{xp_name}/jobs'.format(host=host, xp_name=xp_name))

# --- run experiment ----------------------------------------------------------
max_evals = int(arguments['--max'])

from hyperopt import fmin, tpe, space_eval
best = fmin(
    xp_objective, xp_space,
    trials=trials,
    algo=tpe.suggest, max_evals=max_evals,
    verbose=1)

# --- show results ------------------------------------------------------------
print space_eval(xp_space, best)
