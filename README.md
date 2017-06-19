# __hyppopotamus__ hyper-parameter optimization


## Installation

```bash
$ pip install hyppopotamus
```

## Usage


### Experiment

```bash
$ cat /path/to/experiment.py

# --- experiment unique identifier ------------------
xp_name = 'sin'

# --- hyper-parameters search space -----------------
from hyperopt import hp
xp_space = {'x': hp.uniform('x', -2, 2)}

# --- objective function ----------------------------
def xp_objective(args, **kwargs):
    x = args['x']

    import math
    return math.sin(x)
```

### Hyper-parameter search

```bash
$ hyppopotamus tune /path/to/experiment.py
```

### (parallel) hyper-parameter search

Run master:
```bash
$ ssh master
$ mongod -v -f mongodb.conf
$ export HOST=`hostname`
$ hyppopotamus tune --parallel=master:27017 /path/to/experiment.py
```

See file `mongodb.conf` for an example.

Run first worker,
```bash
$ ssh worker1
$ hyppopotamus work --parallel=master:27017 /path/to/experiment.py
```

On worker 2:
```bash
$ ssh worker2
$ hyppopotamus work --parallel=master:27017 /path/to/experiment.py
```
