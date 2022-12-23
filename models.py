from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, InputLayer, LSTM, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam

MODELS = ['cnn', 'dnn', 'rnn']

CLASSES = {'Classical': 1, 'Electronic': 2, 'Folk': 3, 'Hip-Hop': 4, 'World Music': 5, 'Experimental': 6, 'Pop': 7, 'Rock': 8}
NUM_CLASSES = len(CLASSES)


def cnn(input_shape):
    '''
    Initializes and compiles CNN.
    '''

    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def dnn(input_shape):
    '''
    Initializes and compiles DNN.
    '''

    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def classifier(input_shape):
    '''
    Initializes and compiles a classifier.
    '''

    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    optimizer = Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def rnn(input_shape):
    '''
    Initializes and compiles RNN.
    '''

    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def load(model, input_shape):
    '''
    Initializes and compiles a model.
    '''

    assert model in MODELS

    if model == 'cnn':
        return cnn(input_shape=input_shape)

    if model == 'dnn':
        return dnn(input_shape=input_shape)

    if model == 'rnn':
        return rnn(input_shape=input_shape)
