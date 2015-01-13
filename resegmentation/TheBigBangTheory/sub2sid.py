
# --- define unique name ------------------------------------------------------
xp_name = 'sub2sid'

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
    from pyannote.parser import SRTParser
    from pyannote.core import Segment, Annotation
    from tvd import TheBigBangTheory

    from hyperopt import STATUS_OK
    import numpy as np
    import simplejson as json

    dataset = TheBigBangTheory('/vol/corpora4/tvseries/tvd')
    episodes = dataset.episodes[:6]

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

    NON_SPEECH = set(['non_speech'])

    def get_subtitles(episode):

        from pyannote.features.text.preprocessing import TextPreProcessing
        from pyannote.features.text.tfidf import TFIDF
        from pyannote.algorithms.alignment.transcription import TFIDFAlignment

        preprocessing = TextPreProcessing(
            tokenize=True,    # step 1: tokenization of sentences into words
            lemmatize=True,   # step 2: lemmatization
            stem=True,        # step 3: stemming
            stopwords=True,   # step 4: remove stop-words
            pos_tag=True,
            keep_pos=True,    # step 5: only keep nouns, adjectives, (ad)verbs
            min_length=2)     # step 6: remove stems shorter than 2 letters

        tfidf = TFIDF(preprocessing=preprocessing,
                      binary=True)
        aligner = TFIDFAlignment(tfidf, adapt=True)

        subtitles = SRTParser(split=True, duration=True).read(
            dataset.path_to_subtitles(episode, language='en'))

        transcripts = dataset.get_resource('transcript', episode)

        merged = aligner(transcripts, subtitles,
                         vattribute='speech', hattribute='subtitle')

        # start by adding 'speech' segments
        annotation = Annotation(uri=episode)

        for start_time, end_time, data in merged.edges_iter(data=True):
            if 'speaker' in data:
                if data['speaker'] not in ['SHELDON', 'RAJ', 'PENNY',
                                           'LEONARD', 'HOWARD']:
                    label = 'other'
                else:
                    label = data['speaker'].lower()
                if start_time.anchored and end_time.anchored:
                    annotation[Segment(start_time, end_time)] = label

        # then fill in the gaps with 'non_speech' segments
        extent = Segment(0, dataset.get_episode_duration(episode))
        for gap in annotation.get_timeline().gaps(extent):
            annotation[gap] = 'non_speech'

        return annotation

    subtitles = [get_subtitles(episode) for episode in episodes]

    # --- feature extraction --------------------------------------------------

    zcr = YaafeZCR(sample_rate=16000, block_size=512, step_size=256)
    mfcc = YaafeMFCC(
        sample_rate=16000, block_size=512, step_size=256, **features_param)
    compound = YaafeCompound(
        [zcr, mfcc], sample_rate=16000, block_size=512, step_size=256)

    features = [
        compound(dataset.path_to_audio(episode, language='en'))
        for episode in episodes
    ]

    # --- speech activity detection

    ier = IdentificationErrorRate(collar=0.100)
    attachments = {}

    for e, episode in enumerate(episodes):

        hypothesis = GMMSegmentation.resegment(
            features[e], subtitles[e],
            min_duration=None, constraint=None,
            **algorithm_param)

        reference = groundtruth[e]
        loss = ier(reference.subset(NON_SPEECH, invert=True),
                   hypothesis.subset(NON_SPEECH, invert=True))

        # precomputed loss and hypothesis for current episode
        # will be stored in trial database for later use.
        attachments[episode] = {
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
