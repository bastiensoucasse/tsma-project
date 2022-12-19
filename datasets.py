import pickle

import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = 'data'

DATASETS = ['features', 'melspectro', 'vggish', 'openl3']

TEST_SIZE = 0.2
RANDOM_STATE = 42


def features(hot_one_encode: bool = True, split: bool = True):
    '''
    Retrieves Features dataset as:
        - (ids_train, x_train, y_train), (ids_test, x_test) if split is False.
        - (ids_train, x_train, y_train), (ids_val, x_val, y_val), (ids_test, x_test) if split is True.

    WARNINGS:
        - Features applies only to classifiers (1D data).
    '''

    # Load the dataset.
    features_csv = pd.read_csv(filepath_or_buffer=f'{DATA_DIR}/features.csv', sep=',', iterator=True, chunksize=10000)
    features_csv = pd.concat([chunk for chunk in features_csv])

    # Load the training data.
    train_csv = pd.read_csv(filepath_or_buffer=f'{DATA_DIR}/train.csv', sep=',')
    train_data = pd.merge(train_csv, features_csv, on='track_id')
    ids_train = train_data['track_id'].to_numpy()
    x_train = train_data.drop('genre_id', axis=1).drop('track_id', axis=1)
    for feature in ['fourier_tempogram']:
        for moment in ['min', 'max', 'mean', 'median']:
            for i in range(len(ids_train)):
                x_train[f'{feature}_{moment}'][i] = np.absolute(complex(x_train[f'{feature}_{moment}'][i]))
    x_train = x_train.to_numpy()
    y_train = train_data['genre_id'].to_numpy()

    # Hot-one encode the training labels.
    if hot_one_encode:
        y_train = keras.utils.to_categorical(y_train - 1)

    # Split the training data into training and validation data.
    if split:
        ids_train, ids_val, x_train, x_val, y_train, y_val = train_test_split(ids_train, x_train, y_train, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Load the testing data.
    test_csv = pd.read_csv(filepath_or_buffer=f'{DATA_DIR}/test.csv', sep=',')
    test_data = pd.merge(test_csv, features_csv, on='track_id')
    ids_test = test_data['track_id'].to_numpy()
    x_test = test_data.drop('track_id', axis=1)
    for feature in ['fourier_tempogram']:
        for moment in ['min', 'max', 'mean', 'median']:
            for i in range(len(ids_test)):
                x_test[f'{feature}_{moment}'][i] = np.absolute(complex(x_test[f'{feature}_{moment}'][i]))
    x_test = x_test.to_numpy()

    return ((ids_train, x_train, y_train), (ids_val, x_val, y_val), (ids_test, x_test)) if split else ((ids_train, x_train, y_train), (ids_test, x_test))


def melspectro(hot_one_encode: bool = True, split: bool = True):
    '''
    Retrieves Melspectro dataset as:
        - (ids_train, x_train, y_train), (ids_test, x_test) if split is False.
        - (ids_train, x_train, y_train), (ids_val, x_val, y_val), (ids_test, x_test) if split is True.

    WARNINGS:
        - ids_train is not known (temporary?).
        - hot_one_encode must be True as the raw data is already hot one encoded (temporary?).
        - Melspectro applies only for neural networks (necessarily hot one encoded, temporary?).
    '''

    if not hot_one_encode:
        print('Dataset \'melspetro\' is necessarily hot one encoded.')
        exit(-1)

    # Load the training data.
    x_train = np.array(pickle.load(open(f'{DATA_DIR}/melspectro_x_train.pickle', 'rb')))
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    y_train = np.array(pickle.load(open(f'{DATA_DIR}/melspectro_y_train.pickle', 'rb')))
    ids_train = np.empty(y_train.shape)

    # Split the training data into training and validation data.
    if split:
        ids_train, ids_val, x_train, x_val, y_train, y_val = train_test_split(ids_train, x_train, y_train, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Load the testing data.
    ids_test = np.array(pickle.load(open(f'{DATA_DIR}/melspectro_ids_test.pickle', 'rb')))
    x_test = np.array(pickle.load(open(f'{DATA_DIR}/melspectro_x_test.pickle', 'rb')))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

    return ((ids_train, x_train, y_train), (ids_val, x_val, y_val), (ids_test, x_test)) if split else ((ids_train, x_train, y_train), (ids_test, x_test))


def vggish(hot_one_encode: bool = True, split: bool = True):
    '''
    Retrieves VGGish dataset as:
        - (ids_train, x_train, y_train), (ids_test, x_test) if split is False.
        - (ids_train, x_train, y_train), (ids_val, x_val, y_val), (ids_test, x_test) if split is True.
    '''

    # Load the training dataset.
    train_dataset = pickle.load(open(f'{DATA_DIR}/vggish_train.pickle', 'rb'))
    ids_train = np.array([x[0] for x in train_dataset], dtype=np.int32)
    x_train = np.array([x[2] for x in train_dataset], dtype=np.float32)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    y_train = np.array([x[1] for x in train_dataset], dtype=np.int32)

    # Hot-one encode the training labels.
    if hot_one_encode:
        y_train = keras.utils.to_categorical(y_train - 1)

    # Split the training data into training and validation data.
    if split:
        ids_train, ids_val, x_train, x_val, y_train, y_val = train_test_split(ids_train, x_train, y_train, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Load the testing data.
    test_dataset = pickle.load(open(f'{DATA_DIR}/vggish_test.pickle', 'rb'))
    test_dataset = [(x[0], x[1]) for x in test_dataset if len(x[1]) == 31]
    ids_test = np.array([x[0] for x in test_dataset], dtype=np.int32)
    x_test = np.array([x[1] for x in test_dataset], dtype=np.float32)
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

    return ((ids_train, x_train, y_train), (ids_val, x_val, y_val), (ids_test, x_test)) if split else ((ids_train, x_train, y_train), (ids_test, x_test))


def openl3_dataset(hot_one_encode: bool = True, split: bool = True):
    '''
    Retrieves OpenL3 dataset as:
        - (ids_train, x_train, y_train), (ids_test, x_test) if split is False.
        - (ids_train, x_train, y_train), (ids_val, x_val, y_val), (ids_test, x_test) if split is True.
    '''

    # Load the training dataset.
    train_dataset = pickle.load(open(f'{DATA_DIR}/openl3_train.pickle', 'rb'))
    ids_train = np.array([x[0].replace('.mp3', '') for x in train_dataset], dtype=np.int32)
    x_train = np.array([x[1] for x in train_dataset], dtype=np.float32)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    y_train = pd.read_csv(f'{DATA_DIR}/train.csv').set_index('track_id')
    y_train = np.array([y_train['genre_id'][x] for x in ids_train], dtype=np.int32)

    # Hot-one encode the training labels.
    if hot_one_encode:
        y_train = keras.utils.to_categorical(y_train - 1)

    # Split the training data into training and validation data.
    if split:
        ids_train, ids_val, x_train, x_val, y_train, y_val = train_test_split(ids_train, x_train, y_train, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Load the testing data.
    test_dataset = pickle.load(open(f'{DATA_DIR}/openl3_test.pickle', 'rb'))
    ids_test = np.array([x[0].replace('.mp3', '') for x in test_dataset], dtype=np.int32)
    x_test = np.array([x[1] for x in test_dataset], dtype=np.float32)
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

    return ((ids_train, x_train, y_train), (ids_val, x_val, y_val), (ids_test, x_test)) if split else ((ids_train, x_train, y_train), (ids_test, x_test))


def load(dataset: str, hot_one_encode: bool = True, split: bool = True):
    '''
    Retrieves a dataset as:
        - (ids_train, x_train, y_train), (ids_test, x_test) if split is False.
        - (ids_train, x_train, y_train), (ids_val, x_val, y_val), (ids_test, x_test) if split is True.
    '''

    assert dataset in DATASETS

    if dataset == 'melspectro':
        return melspectro(hot_one_encode=hot_one_encode, split=split)

    if dataset == 'features':
        return features(hot_one_encode=hot_one_encode, split=split)

    if dataset == 'vggish':
        return vggish(hot_one_encode=hot_one_encode, split=split)

    if dataset == 'openl3':
        return openl3_dataset(hot_one_encode=hot_one_encode, split=split)


if __name__ == '__main__':
    print(f'Datasets Tests')
    for d in DATASETS:
        print(f'\nProcessing \'{d}\'…')
        (ids_train, x_train, y_train), (ids_val, x_val, y_val), (ids_test, x_test) = load(d, True, True)
        print(f'Train data shape: {x_train.shape} and {y_train.shape}.')
        print(f'Validation data shape: {x_val.shape} and {y_val.shape}.')
        print(f'Test data shape: {x_test.shape}.')
