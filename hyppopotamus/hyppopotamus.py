"""Hyppopotamus


Usage:
  hyppopotamus tune  (--mongo=<host> | --pickle=<file.pkl) --work-dir=<workdir> [--luigi=<host>] [--max-evals=<number>] <experiment.py>
  hyppopotamus rerun (--mongo=<host> | --pickle=<file.pkl) --work-dir=<workdir> [--luigi=<host>] <experiment.py>
  hyppopotamus best  (--mongo=<host> | --pickle=<file.pkl) <experiment.py>
  hyppopotamus plot  [options] (--mongo=<host> | --pickle=<file.pkl) <experiment.py> <output_dir>
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

Plotting (plot):
  --y-min=<min>             [default: 0]
  --y-max=<max>             [default: 1]

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
import numpy as np
import sys


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


def rerun(xp_name, xp_space, xp_objective,
          mongo_host=None, trials_pkl=None,
          work_dir=None, luigi_host=None):

    # --- load experiment trials ---------------------------------------------
    if trials_pkl is not None:
        with open(trials_pkl, 'r') as fp:
            trials = pickle.load(fp)

    if mongo_host is not None:
        TEMPLATE = 'mongo://{host}/{xp_name}/jobs'
        url = TEMPLATE.format(host=mongo_host, xp_name=xp_name)
        trials = hyperopt.mongoexp.MongoTrials(url)

    trial = trials.best_trial
    params = {key: value[0] for key, value in trial['misc']['vals'].items()}
    params = space_eval(xp_space, params)

    pprint(params)

    xp_objective(params, luigi_host=luigi_host, work_dir=work_dir)


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


def plot(output_dir, xp_name, xp_space, y_min=0., y_max=1., mongo_host=None, trials_pkl=None):

    import matplotlib
    matplotlib.use('pdf')
    from matplotlib import pyplot as plt

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
        loss.append(result.get('loss', np.inf))
        true_loss.append(result.get('true_loss', np.inf))
        loss_variance.append(result.get('loss_variance', np.inf))
        true_loss_variance.append(result.get('true_loss_variance', np.inf))

        trial_params = {key: value[0] for key, value in trial['misc']['vals'].items()}
        trial_params = space_eval(xp_space, trial_params)

        for name in params:
            param_value = trial_params[name]
            params[name].append(param_value)

    # --- loss & true loss ----------------------------------------------------
    fig, ax = plt.subplots()

    LABEL = '{subset} ({loss:.3f})'
    label = LABEL.format(subset='dev', loss=np.min(loss))
    ax.plot(np.minimum.accumulate(loss), label=label)

    # compute (true) true loss, as the performance on the test
    # by the best performing system on the dev set
    _true_loss = []
    best_loss = np.inf
    for i, _loss in enumerate(loss):
        if _loss < best_loss:
            _true_loss.append(true_loss[i])
            best_loss = _loss
        else:
            _true_loss.append(_true_loss[-1])

    label = LABEL.format(subset='test', loss=true_loss[np.argmin(loss)])
    ax.plot(_true_loss, label=label)

    # axes, legend and title
    ax.set_ylim(y_min, y_max)
    ax.legend()
    TITLE = '{xp_name} ({n:d} trials)'
    ax.set_title(TITLE.format(xp_name=xp_name, n=len(loss)))

    # save to file
    TEMPLATE = '{output_dir}/{xp_name}.loss.pdf'
    path = TEMPLATE.format(output_dir=output_dir, xp_name=xp_name)
    fig.savefig(path)

    # --- params --------------------------------------------------------------
    for name in params:

        try:
            fig, ax = plt.subplots()

            ax.plot(params[name], '.')
            m, M = np.min(params[name]), np.max(params[name])
            ax.set_ylim(m - 0.1 * (M-m), M + 0.1 * (M-m))

            TITLE = '{xp_name} - {param}'
            ax.set_title(TITLE.format(xp_name=xp_name, param=name))

            # save to file
            TEMPLATE = '{output_dir}/{xp_name}.{param}.pdf'
            path = TEMPLATE.format(
                output_dir=output_dir, xp_name=xp_name, param=name)
            fig.savefig(path)

        except Exception as e:
            sys.stderr.write('Cannot plot "{param}" parameter.\n'.format(param=name))




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

    if arguments['rerun']:

        work_dir = arguments['--work-dir']
        luigi_host = arguments['--luigi']

        rerun(xp_name, xp_space, xp_objective,
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
        y_min = float(arguments['--y-min'])
        y_max = float(arguments['--y-max'])
        output_dir = arguments['<output_dir>']
        plot(output_dir, xp_name, xp_space,
             y_min=y_min, y_max=y_max,
             mongo_host=mongo_host,
             trials_pkl=trials_pkl)
