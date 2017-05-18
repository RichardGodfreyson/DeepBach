import pickle
import numpy as np

from data_utils import BACH_DATASET, START_SYMBOL, END_SYMBOL, to_onehot, indexed_chorale_to_score
from keras.engine import Input, Model
from keras.layers import Dense, Activation, Dropout
from keras.models import load_model

SOP_INDEX = 0


def minibach(input_size, output_size, hidden_size):
    features = Input((input_size,))
    preds = Dropout(0.2)(features)
    preds = Dense(hidden_size)(features)
    preds = Activation('relu')(preds)
    preds = Dropout(0.5)(preds)
    preds = Dense(output_size)(preds)
    preds = Activation('sigmoid')(preds)

    return Model(inputs=features, outputs=preds)


def features_and_target(chorale, time_index,
                        timesteps, num_pitches,
                        num_voices):
    """
    
    :param chorale: time major 
    :param time_index: 
    :param timesteps: 
    :param num_pitches: 
    :param num_voices: 
    :return: 
    """
    features = np.array([to_onehot(time_slice, num_pitches[SOP_INDEX])
                         for time_slice in chorale[time_index: time_index + timesteps, SOP_INDEX]
                         ])
    # (time, num_pitches)
    features = np.reshape(features, (timesteps * num_pitches[SOP_INDEX],))
    # (time * num_pitches)

    target = [to_onehot(time_slice, num_pitches[voice_index])
              for voice_index in range(1, num_voices)
              for time_slice in chorale[time_index: time_index + timesteps, voice_index]]
    target = np.concatenate(target)

    # target[(i * timesteps * num_pitches +  j * num_pitches + k)] = voice i, timestep j, pitch k
    return features, target


def reconstruct(features, target, num_pitches, timesteps, num_voices):
    """
    
    :param features: 
    :param target: 
    :param num_pitches: 
    :param timesteps: 
    :param num_voices: 
    :return: voice major chorale 
    """
    sop = np.reshape(features, (timesteps, num_pitches[SOP_INDEX]))
    sop = np.array([np.argmax(time_slice) for time_slice in sop])

    other_voices = []

    offset = 0
    for voice_index in range(1, num_voices):
        voice_length = num_pitches[voice_index] * timesteps

        new_voice = target[offset: voice_length + offset]
        new_voice = np.reshape(new_voice, (timesteps, num_pitches[voice_index]))
        new_voice = np.array([np.argmax(time_slice) for time_slice in new_voice])

        other_voices.append(new_voice)
        offset += voice_length

    assert offset == len(target)

    return np.stack([sop] + other_voices, axis=0)


def generator_pianoroll(batch_size, timesteps,
                        phase='train', percentage_train=0.8, pickled_dataset=BACH_DATASET):
    """
     Returns a generator of
            (left_features,
            central_features,
            right_features,
            beats,
            metas,
            labels,
            fermatas) tuples

            where fermatas = (fermatas_left, central_fermatas, fermatas_right)
    """

    X, X_metadatas, voice_ids, index2notes, note2indexes, metadatas = pickle.load(open(pickled_dataset, 'rb'))
    num_pitches = list(map(lambda x: len(x), index2notes))
    num_voices = len(voice_ids)
    # Set chorale_indices
    if phase == 'train':
        chorale_indices = np.arange(int(len(X) * percentage_train))
    if phase == 'test':
        chorale_indices = np.arange(int(len(X) * percentage_train), len(X))
    if phase == 'all':
        chorale_indices = np.arange(int(len(X)))

    features = []
    targets = []
    batch = 0

    while True:
        chorale_index = np.random.choice(chorale_indices)
        extended_chorale = np.transpose(X[chorale_index])

        padding_dimensions = (timesteps,) + extended_chorale.shape[1:]

        start_symbols = np.array(list(map(lambda note2index: note2index[START_SYMBOL], note2indexes)))
        end_symbols = np.array(list(map(lambda note2index: note2index[END_SYMBOL], note2indexes)))

        extended_chorale = np.concatenate((np.full(padding_dimensions, start_symbols),
                                           extended_chorale,
                                           np.full(padding_dimensions, end_symbols)),
                                          axis=0)

        chorale_length = len(extended_chorale)

        time_index = np.random.randint(0, chorale_length - timesteps)

        feature, target = features_and_target(chorale=extended_chorale, time_index=time_index,
                                              timesteps=timesteps, num_pitches=num_pitches,
                                              num_voices=num_voices)

        features.append(feature)
        targets.append(target)

        batch += 1

        # if there is a full batch
        if batch == batch_size:
            next_element = (
                np.array(features, dtype=np.float32),
                np.array(targets, dtype=np.float32))

            yield next_element

            batch = 0

            features = []
            targets = []


if __name__ == '__main__':
    batch_size = 128
    timesteps = 16
    gen_train = generator_pianoroll(batch_size, timesteps=timesteps, phase='train')
    gen_test = generator_pianoroll(batch_size, timesteps=timesteps, phase='test')

    features, target = next(gen_train)
    input_size = features.shape[-1]
    output_size = target.shape[-1]


    # Choose first line if model does not exist
    # model = minibach(input_size, output_size, hidden_size=1280)
    model = load_model('models/minibach.h5')

    # Train model, comment the following lines to directly generate examples
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit_generator(generator=gen_train, steps_per_epoch=1000,
                        epochs=20,
                        validation_data=gen_test,
                        validation_steps=20)
    model.save(filepath='models/minibach.h5')



    # Generate example
    # load dataset
    X, X_metadatas, voice_ids, index2notes, note2indexes, metadatas = pickle.load(open(BACH_DATASET, 'rb'))
    num_pitches = list(map(lambda x: len(x), index2notes))
    num_voices = len(voice_ids)
    del X, X_metadatas

    # pick up one example
    features, target = next(gen_test)
    features = features[0]
    target = target[0]

    # show original chorale
    reconstructed_chorale = reconstruct(features, target, num_pitches, timesteps, num_voices)
    score = indexed_chorale_to_score(reconstructed_chorale, BACH_DATASET)
    score.show()

    # show predicted chorale
    predictions = model.predict(np.array([features]), batch_size=1)[0]
    reconstructed_predicted_chorale = reconstruct(features, predictions, num_pitches, timesteps, num_voices)
    score = indexed_chorale_to_score(reconstructed_predicted_chorale, BACH_DATASET)
    score.show()
