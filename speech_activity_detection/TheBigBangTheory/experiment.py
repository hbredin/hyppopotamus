
# --- define unique name ------------------------------------------------------
xp_name = 'tbbt_sad'
# xp_name = 'tune_sad'

# --- define hyper-parameters search space ------------------------------------

from hyperopt import hp
trueOrFalse = [True, False]
xp_space = [
    hp.choice('e', trueOrFalse),
    hp.quniform('coefs', 11, 15, 1),
    hp.choice('De', trueOrFalse),
    hp.choice('DDe', trueOrFalse),
    hp.choice('D', trueOrFalse),
    hp.choice('DD', trueOrFalse),
    2 ** hp.quniform('n_components', 0, 10, 1),  # from 1 to 1024 gaussians
    hp.choice('calibration', [None, 'isotonic', 'naive_bayes']),
    hp.choice('equal_priors', trueOrFalse)
]


# --- define experiment objective function ------------------------------------

def xp_objective(args):
    e, coefs, De, DDe, D, DD, n_components, calibration, equal_priors = args

    from pyannote.features.audio.yaafe import \
        YaafeZCR, YaafeMFCC, YaafeCompound
    from pyannote.metrics.identification import IdentificationErrorRate
    from pyannote.algorithms.segmentation.hmm import GMMSegmentation
    from sklearn.cross_validation import LeaveOneOut
    from tvd import TheBigBangTheory

    print args

    n_components = int(n_components)
    coefs = int(coefs)

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
    mfcc = YaafeMFCC(e=e, coefs=coefs, De=De, DDe=DDe, D=D, DD=DD)
    compound = YaafeCompound(
        [zcr, mfcc], sample_rate=16000, block_size=512, step_size=256)

    features = [
        compound(dataset.path_to_audio(episode, language='en'))
        for episode in episodes
    ]

    # --- speech activity detection

    ier = IdentificationErrorRate(collar=0.100)

    for trn, tst in LeaveOneOut(n_episodes):

        tst = tst[0]

        segmenter = GMMSegmentation(
            n_jobs=2, n_iter=20, lbg=True,  # 2x faster, 5x faster, much faster
            n_components=n_components,
            calibration=calibration,
            equal_priors=equal_priors)

        segmenter.fit(
            [features[episode] for episode in trn],
            [groundtruth[episode] for episode in trn]
        )

        hypothesis = segmenter.predict(features[tst])
        reference = groundtruth[tst]
        res = ier(reference, hypothesis)

        print episodes[tst], res

    res = abs(ier)
    print '==>', res

    return abs(ier)
