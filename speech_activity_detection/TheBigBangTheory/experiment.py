
# --- define unique name ------------------------------------------------------
xp_name = 'sad'

# --- define hyper-parameters search space ------------------------------------

from hyperopt import hp
trueOrFalse = [True, False]
xp_space = {
    'features': {
        'e': hp.choice('e', trueOrFalse),
        'De': hp.choice('De', trueOrFalse),
        'DDe': hp.choice('DDe', trueOrFalse),
        'coefs': hp.quniform('coefs', 11, 15, 2),
        'D': hp.choice('D', trueOrFalse),
        'DD': hp.choice('DD', trueOrFalse),
    },
    'algorithm': {
        'n_components': 2 ** hp.quniform('n_components', 0, 10, 1),
        'calibration': hp.choice('calibration',
                                 [None, 'isotonic', 'naive_bayes']),
        'equal_priors': hp.choice('equal_priors', trueOrFalse)
    }
}


# --- define experiment objective function ------------------------------------

def xp_objective(parameters):

    features_param = parameters['features']
    algorithm_param = parameters['algorithm']
    algorithm_param['n_components'] = int(algorithm_param['n_components'])

    from pyannote.features.audio.yaafe import \
        YaafeZCR, YaafeMFCC, YaafeCompound
    from pyannote.metrics.identification import IdentificationErrorRate
    from pyannote.algorithms.segmentation.hmm import GMMSegmentation
    from sklearn.cross_validation import LeaveOneOut
    from tvd import TheBigBangTheory

    from hyperopt import STATUS_OK
    import numpy as np
    import simplejson as json

    dataset = TheBigBangTheory('/vol/corpora4/tvseries/tvd')
    episodes = dataset.episodes[:6]
    n_episodes = len(episodes)

    # --- groundtruth ---------------------------------------------------------

    mapping = {
        'music_titlesong': 'non_speech',
        'silence': 'non_speech',
        'sound_laughclap': 'non_speech',
        'sound_laugh': 'non_speech',
        'sound_other': 'non_speech',
        'speech_howard': 'speech',
        'speech_leonard': 'speech',
        'speech_other': 'speech',
        'speech_overlapping': 'speech',
        'speech_penny': 'speech',
        'speech_raj': 'speech',
        'speech_sheldon': 'speech',
    }

    groundtruth = [
        dataset.get_resource('speaker', episode).translate(mapping)
        for episode in episodes
    ]

    # --- feature extraction --------------------------------------------------

    zcr = YaafeZCR()
    mfcc = YaafeMFCC(**features_param)
    compound = YaafeCompound(
        [zcr, mfcc], sample_rate=16000, block_size=512, step_size=256)

    features = [
        compound(dataset.path_to_audio(episode, language='en'))
        for episode in episodes
    ]

    # --- speech activity detection

    ier = IdentificationErrorRate(collar=0.100)
    attachments = {}

    for trn, tst in LeaveOneOut(n_episodes):

        tst = tst[0]

        segmenter = GMMSegmentation(
            n_jobs=1,  # n_jobs > 1 will fail (not sure why)
            n_iter=20, lbg=True,
            **algorithm_param)

        segmenter.fit(
            [features[episode] for episode in trn],
            [groundtruth[episode] for episode in trn]
        )

        hypothesis = segmenter.predict(features[tst])
        reference = groundtruth[tst]
        loss = ier(reference, hypothesis)

        # precomputed loss and hypothesis for current episode
        # will be stored in trial database for later use.
        attachments[episodes[tst]] = {
            'loss': loss,
            'hypothesis': hypothesis}

    status = STATUS_OK
    loss = abs(ier)
    loss_variance = np.var([x['loss'] for x in attachments.values()])

    # dictionary of key-value pairs whose keys are short strings (str(episode))
    # and whose values are potentially long strings (json.dumps)
    attachments = {str(episode): json.dumps(attachment, for_json=True)
                   for episode, attachment in attachments.iteritems()}

    return {'status': status,
            'loss': loss,
            'loss_variance': loss_variance,
            'attachments': attachments}
