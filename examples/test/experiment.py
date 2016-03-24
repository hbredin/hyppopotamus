
# --- define unique name ------------------------------------------------------
xp_name = 'sin'

# --- define hyper-parameters search space ------------------------------------

from hyperopt import hp
xp_space = hp.uniform('x', -2, 2)


# --- define experiment objective function ------------------------------------

def xp_objective(args):
    x = args

    import math
    return math.sin(x)
