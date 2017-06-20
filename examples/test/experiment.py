
# --- define unique name ------------------------------------------------------
xp_name = 'sin'

# --- define hyper-parameters search space ------------------------------------

from hyperopt import hp
xp_space = {'x': hp.uniform('x', -2, 2)}


# --- define experiment objective function ------------------------------------

def xp_objective(args, **kwargs):
    x = args['x']

    import math
    return math.sin(x)

    # return {'loss': math.sin(x),
    #         'loss_variance': loss_variance,
    #         'status': status,
    #         'attachments': attachments}
