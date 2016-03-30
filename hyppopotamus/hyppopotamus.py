"""Hyppopotamus


Usage:
  hyppopotamus tune [options] (--mongo=<host> | --pickle=<file.pkl) <experiment.py>
  hyppopotamus best (--mongo=<host> | --pickle=<file.pkl) <experiment.py>
  hyppopotamus plot (--mongo=<host> | --pickle=<file.pkl) <experiment.py>
  hyppopotamus reset (--mongo=<host> | --pickle=<file.pkl) <experiment.py>
  hyppopotamus (-h | --help)
  hyppopotamus --version

General options:

  <experiment.py>           Path to Hyppopotamus experiment file.
  --mongo=<host>            Parallel search using this MongoDB host.
  --pickle=<file.pkl>       Sequential search using this pickled trial file.

  -h --help                 Show this screen.
  --version                 Show version.
  --verbose                 Show processing progress.

Perform hyper-parameters tuning (tune):

  --max-evals=<number>      Allow up to this many evaluations [default: 100].
  --work-dir=<workdir>      Add <workdir> to set of parameters.
  --luigi=<host>            Add <luigi_host> to set of parameters.

"""

from __future__ import print_function

import pickle
import functools
import hyperopt
import hyperopt.mongoexp
import pymongo
from hyperopt import fmin, tpe, space_eval
from hyperopt import STATUS_OK, STATUS_FAIL, STATUS_NEW, STATUS_RUNNING
from docopt import docopt
from pprint import pprint



def tune(xp_name, xp_space, xp_objective,
         max_evals=100,
         mongo_host=None, trials_pkl=None,
         work_dir=None, luigi_host=None):

    # --- load experiment trials ---------------------------------------------
    if mongo_host is None:
        try:
            with open(trials_pkl, 'r') as fp:
                trials = pickle.load(fp)
        except:
            trials = hyperopt.Trials()

    else:
        TEMPLATE = 'mongo://{host}/{xp_name}/jobs'
        url = TEMPLATE.format(host=mongo_host, xp_name=xp_name)
        trials = hyperopt.mongoexp.MongoTrials(url)

    xp_objective = functools.partial(
        xp_objective, luigi_host=luigi_host, work_dir=work_dir)

    # --- actual hyper-parameters optimization --------------------------------
    best = hyperopt.fmin(
        xp_objective, xp_space,
        trials=trials,
        algo=hyperopt.tpe.suggest,
        max_evals=max_evals,
        verbose=1)

    # --- show results -------------------------------------------------------
    print(hyperopt.space_eval(xp_space, best))

    # --- pickle trials-------------------------------------------------------
    if trials_pkl is not None:
        with open(trials_pkl, 'w') as fp:
            pickle.dump(trials, fp)


def best(xp_name, xp_space, mongo_host=None, trials_pkl=None):

    # --- load experiment trials ---------------------------------------------
    if trials_pkl is not None:
        with open(trials_pkl, 'r') as fp:
            trials = pickle.load(fp)

    if mongo_host is not None:
        TEMPLATE = 'mongo://{host}/{xp_name}/jobs'
        url = TEMPLATE.format(host=mongo_host, xp_name=xp_name)
        trials = hyperopt.mongoexp.MongoTrials(url)

    print('#> BEST LOSS')
    best = trials.best_trial
    result = dict(best['result'])
    result.pop('status')
    pprint(result)

    print('#> BEST PARAMETERS')
    pprint(space_eval(xp_space, trials.argmin))


def reset(xp_name, mongo_host=None, trials_pkl=None):

    # create empty trials and save it to file
    if trials_pkl is not None:
        trials = hyperopt.Trials()
        with open(trials_pkl, 'w') as fp:
            pickle.dump(trials.fp)

    # empty mongo database
    if mongo_host is not None:
        client = pymongo.MongoClient(host=mongo_host, port=None)
        client.drop_database(xp_name)
        client.close()


def plot(xp_name, xp_space, mongo_host=None, trials_pkl=None):

    colors = {
        STATUS_NEW: 'k',
        STATUS_RUNNING: 'g',
        STATUS_OK: 'b',
        STATUS_FAIL: 'r'}

    # --- load experiment trials ---------------------------------------------
    if trials_pkl is not None:
        with open(trials_pkl, 'r') as fp:
            trials = pickle.load(fp)

    if mongo_host is not None:
        TEMPLATE = 'mongo://{host}/{xp_name}/jobs'
        url = TEMPLATE.format(host=mongo_host, xp_name=xp_name)
        trials = hyperopt.mongoexp.MongoTrials(url)

    # get list of hyper-parameters from first trial
    trial = trials.trials[0]
    params = {name: [] for name in trial['misc']['vals']}

    status = []
    loss = []
    loss_variance = []
    true_loss = []
    true_loss_variance = []

    for t, trial in enumerate(trials.trials):

        result = trial['result']
        status.append(result.get('status'))
        loss.append(result.get('loss', None))
        true_loss.append(result.get('true_loss', None))
        loss_variance.append(result.get('loss_variance', None))
        true_loss_variance.append(result.get('true_loss_variance', None))

        trial_params = {key: value[0] for key, value in trial['misc']['vals'].items()}
        trial_params = space_eval(xp_space, trial_params)

        for name in params:
            param_value = trial_params[name]
            params[name].append(param_value)

    pprint(loss)
    pprint(true_loss)
    pprint(params)


if __name__ == '__main__':

    # parse command line arguments
    arguments = docopt(__doc__)

    # --- load experiment objective function and search space ----------------
    experiment_py = arguments['<experiment.py>']
    execfile(experiment_py)

    mongo_host = arguments['--mongo']
    trials_pkl = arguments['--pickle']

    if arguments['tune']:

        max_evals = int(arguments['--max-evals'])
        work_dir = arguments['--work-dir']
        luigi_host = arguments['--luigi']

        tune(xp_name, xp_space, xp_objective,
             max_evals=max_evals,
             mongo_host=mongo_host,
             trials_pkl=trials_pkl,
             work_dir=work_dir,
             luigi_host=luigi_host)

    if arguments['best']:
        best(xp_name, xp_space,
             mongo_host=mongo_host,
             trials_pkl=trials_pkl)

    if arguments['reset']:
        reset(xp_name, mongo_host=mongo_host, trials_pkl=trials_pkl)

    if arguments['plot']:
        plot(xp_name, xp_space,
             mongo_host=mongo_host,
             trials_pkl=trials_pkl)
