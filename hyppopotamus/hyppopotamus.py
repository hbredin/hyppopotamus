#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

"""Hyppopotamus -- hyper-parameter optimization

Usage:
  hyppopotamus tune [--parallel=<host>] [--max-evals=<number>] <experiment.py>
  hyppopotamus work --parallel=<host> <experiment.py>
  hyppopotamus best [--parallel=<host>] [--run] <experiment.py>
  hyppopotamus plot [options] [--parallel=<host>] <experiment.py> <output_dir>
  hyppopotamus reset [--parallel=<host>] <experiment.py>
  hyppopotamus (-h | --help)
  hyppopotamus --version

Options:
  <experiment.py>           Path to experiment file.
  --parallel=<host>         Parallel search using this MongoDB host.

  -h --help                 Show this screen.
  --version                 Show version.
  --verbose                 Show processing progress.

  --max-evals=<number>      Allow up to this many evaluations [default: 100].

Plotting (plot):
  --y-min=<min>             [default: 0]
  --y-max=<max>             [default: 1]

"""

from __future__ import print_function

import sys
import pickle
import os.path
import functools

from pymongo import MongoClient
from hyperopt.mongoexp import MongoTrials, MongoJobs, MongoWorker
from hyperopt import fmin, tpe, space_eval, Trials
from hyperopt import STATUS_OK, STATUS_FAIL, STATUS_NEW, STATUS_RUNNING

import numpy as np
from docopt import docopt
from pprint import pprint
from datetime import datetime

def setup_xp(experiment_py):
    """Load xp_{name | space | objective} global variables

    Parameters
    ----------
    experiment_py : str
        Path to experiment file.
    """
    with open(experiment_py) as fp:
        code = compile(fp.read(), experiment_py, 'exec')
        exec(code, globals())

def run_xp(experiment_py, args):
    """Run experiment

    Parameters
    ----------
    experiment_py : str
        Path to the experiment file.
    args :
        hyperopt-generated arguments

    Returns
    -------
    objective : float
        Value of the objective function
    """

    setup_xp(experiment_py)
    return xp_objective(args)

def setup_trials(experiment_py, host=None):
    """Get Trials instance

    Parameters
    ----------
    experiment_py : str
        Path to the experiment file.
    host : str, optional
        URL to MongoDB instance.

    Returns
    -------
    trials
    """

    if host is None:
        trials_pkl = os.path.join(os.path.dirname(experiment_py),
                                  'trials.pkl')
        try:
            with open(trials_pkl, 'rb') as fp:
                trials = pickle.load(fp)
        except FileNotFoundError:
            trials = Trials()

        return trials

    setup_xp(experiment_py)
    url = 'mongo://{host}/{xp_name}/jobs'.format(host=host,
                                                 xp_name=xp_name)
    return MongoTrials(url)


def tune(experiment_py, max_evals=100, host=None, trials_pkl=None):

    setup_xp(experiment_py)

    trials = setup_trials(experiment_py, host=host)

    xp_objective = functools.partial(run_xp, experiment_py)

    best = fmin(xp_objective, xp_space, trials=trials,
        algo=tpe.suggest, max_evals=max_evals, verbose=1)

    pprint(space_eval(xp_space, best))

    if host is None:
        trials_pkl = os.path.join(os.path.dirname(experiment_py),
                                  'trials.pkl')
        with open(trials_pkl, 'wb') as fp:
            pickle.dump(trials, fp)

def work(experiment_py, host):

    setup_xp(experiment_py)

    mongo = 'mongo://{host}/{xp_name}/jobs'.format(host=host, xp_name=xp_name)
    job = MongoJobs.new_from_connection_str(mongo)
    poll_interval = 5.
    workdir = os.path.dirname(experiment_py)
    worker = MongoWorker(job, poll_interval, workdir=workdir)

    while True:
        worker.run_one(erase_created_workdir=True)
        # TODO log time when


def best(experiment_py, host=None, run=False):

    trials = setup_trials(experiment_py, host=host)

    n_trials = len([t for t in trials.trials
                      if t['result']['status'] == STATUS_OK])

    if n_trials == 0:
        print('No completed trials yet.')
        return

    print('#> BEST LOSS (out of {n} trials)'.format(n=n_trials))
    best = trials.best_trial

    result = dict(best['result'])
    for report in ['loss', 'loss_variance', 'true_loss', 'true_loss_variance']:
        if report not in result:
            continue
        TEMPLATE = '{report}: {value:g}'
        print(TEMPLATE.format(report=report, value=result[report]))

    print('#> BEST PARAMETERS')
    setup_xp(experiment_py)
    pprint(space_eval(xp_space, trials.argmin))

    if run:
        params = {key: value[0] for key, value in best['misc']['vals'].items()}
        params = space_eval(xp_space, params)
        pprint(xp_objective(params))


def reset(experiment_py, host=None):
    """Reset experiment"""

    if host is None:
        trials_pkl = os.path.join(os.path.dirname(experiment_py), 'trials.pkl')
        with open(trials_pkl, 'wb') as fp:
            pickle.dump(Trials(), fp)

    else:
        setup_xp(experiment_py)
        client = MongoClient(host=host, port=None)
        client.drop_database(xp_name)
        client.close()

# def key_func(trial):
#     """Key to use with `sorted`
#
#     Use it to sort trials in the following order:
#     - finished trials first, then running trials, then new trials
#     - within finished trials, sort by end time
#     - within running trials, sort by start time
#     - within new trials, sort by creation time
#     """
#
#     _order = {STATUS_OK: 1, STATUS_FAIL: 1,
#               STATUS_RUNNING: 2, STATUS_NEW: 3}
#     status_order = _order[trial['result']['status']]
#     refresh_time = trial['refresh_time']
#     refresh_time = datetime.now() if refresh_time is None else refresh_time
#     return status_order, refresh_time
#
# def plot(output_dir, experiment_py, y_min=0., y_max=1., host=None):
#
#     import matplotlib
#     matplotlib.use('pdf')
#     from matplotlib import pyplot as plt
#     from matplotlib.dates import DayLocator, HourLocator, DateFormatter
#
#     COLORS = {
#         STATUS_NEW: 'b',
#         STATUS_RUNNING: 'b',
#         STATUS_OK: 'g',
#         STATUS_FAIL: 'r'}
#
#     trials = setup_trials(experiment_py, host=host)
#
#     # get list of hyper-parameters from first trial
#     trial = trials.trials[0]
#     params = {name: [] for name in trial['misc']['vals']}
#
#     status = []
#     colors = []
#     loss = []
#     loss_variance = []
#     true_loss = []
#     true_loss_variance = []
#     time = []
#
#     # sort trials by (end_time, start_time)
#     trials = sorted(trials.trials, key=key_func)
#
#     setup_xp(experiment_py)
#
#     for t, trial in enumerate(trials):
#
#         result = trial['result']
#         status.append(result.get('status'))
#         colors.append(COLORS[status[-1]])
#
#         if 'loss' in result:
#             loss.append(result.get('loss'))
#             true_loss.append(result.get('true_loss'))
#             loss_variance.append(result.get('loss_variance'))
#             true_loss_variance.append(result.get('true_loss_variance'))
#             time.append(trial['refresh_time'])
#
#         trial_params = {key: value[0] for key, value in trial['misc']['vals'].items()}
#         trial_params = space_eval(xp_space, trial_params)
#         for name in params:
#             param_value = trial_params[name]
#             params[name].append(param_value)
#
#     # --- loss & true loss ----------------------------------------------------
#     if len(loss) > 0:
#
#         fig, ax = plt.subplots()
#
#         LABEL = '{subset} ({loss:.3f})'
#         label = LABEL.format(subset='dev', loss=np.min(loss))
#         ax.plot_date(time, np.minimum.accumulate(loss), fmt='-', xdate=True, label=label)
#
#         # compute (true) true loss, as the performance on the test
#         # by the best performing system on the dev set
#         _true_loss = []
#         best_loss = np.inf
#         for i, _loss in enumerate(loss):
#             if _loss < best_loss:
#                 _true_loss.append(true_loss[i])
#                 best_loss = _loss
#             else:
#                 _true_loss.append(_true_loss[-1])
#
#         label = LABEL.format(subset='test', loss=true_loss[np.argmin(loss)])
#         ax.plot_date(time, _true_loss, fmt='-', xdate=True, label=label)
#
#         # years =    # every year
#         # months = MonthLocator()  # every month
#         # yearsFmt = DateFormatter('%Y-%M-%D')
#
#         ax.xaxis.set_major_locator(DayLocator())
#         ax.xaxis.set_major_formatter(DateFormatter('%Y/%m/%d'))
#         ax.xaxis.set_minor_locator(HourLocator())
#         ax.autoscale_view()
#
#         # axes, legend and title
#         ax.set_ylim(y_min, y_max)
#         ax.legend()
#         TITLE = '{xp_name} ({n:d} trials)'
#         ax.set_title(TITLE.format(xp_name=xp_name, n=len(loss)))
#
#         # save to file
#         TEMPLATE = '{output_dir}/{xp_name}.loss.pdf'
#         path = TEMPLATE.format(output_dir=output_dir, xp_name=xp_name)
#         fig.savefig(path)
#
#     # --- params --------------------------------------------------------------
#     for name in params:
#
#         fig, ax = plt.subplots()
#
#         try:
#             ax.scatter(range(len(params[name])), params[name], s=50, lw=0, c=colors, marker=u'o')
#             m, M = np.min(params[name]), np.max(params[name])
#             ax.set_ylim(m - 0.1 * (M-m), M + 0.1 * (M-m))
#
#         except Exception as e:
#
#             fig, ax = plt.subplots()
#
#             unique, unique_indices, unique_inverse = np.unique(
#                 params[name], return_index=True, return_inverse=True)
#             ax.scatter(range(len(unique_inverse)), unique_inverse, s=50, alpha=0.5, lw=0, c=colors, marker=u'o')
#             m, M = np.min(unique_inverse), np.max(unique_inverse)
#             ax.set_yticks(range(len(unique)))
#             ax.set_yticklabels(unique)
#             ax.set_ylim(m - 0.5 * (M-m), M + 0.5 * (M-m))
#
#         TITLE = '{xp_name} - {param}'
#         ax.set_title(TITLE.format(xp_name=xp_name, param=name))
#         plt.tight_layout()
#
#         # save to file
#         TEMPLATE = '{output_dir}/{xp_name}.{param}.pdf'
#         path = TEMPLATE.format(
#             output_dir=output_dir, xp_name=xp_name, param=name)
#         fig.savefig(path)


def main():

    arguments = docopt(__doc__)

    # get experiment absolute path
    experiment_py = arguments['<experiment.py>']
    experiment_py = os.path.abspath(experiment_py)

    host = arguments['--parallel']

    if arguments['tune']:
        max_evals = int(arguments['--max-evals'])
        tune(experiment_py, max_evals=max_evals, host=host)

    if arguments['work']:
        work(experiment_py, host)

    if arguments['best']:
        run = arguments['--run']
        best(experiment_py, host=host, run=run)

    if arguments['reset']:
        reset(experiment_py, host=host)

    # if arguments['plot']:
    #     y_min = float(arguments['--y-min'])
    #     y_max = float(arguments['--y-max'])
    #     output_dir = arguments['<output_dir>']
    #     plot(output_dir, experiment_py, host=host,
    #          y_min=y_min, y_max=y_max)
