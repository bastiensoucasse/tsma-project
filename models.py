from keras.layers import BatchNormalization, Conv2D, Dense, Flatten, InputLayer, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import RMSprop

MODELS = ['cnn', 'dnn']

CLASSES = {'Classical': 1, 'Electronic': 2, 'Folk': 3, 'Hip-Hop': 4, 'World Music': 5, 'Experimental': 6, 'Pop': 7, 'Rock': 8}
NUM_CLASSES = len(CLASSES)


def cnn(input_shape=None) -> Sequential:
    '''
    Initializes and compiles CNN.
    '''

    model = Sequential()
    if input_shape is not None:
        model.add(InputLayer(input_shape=input_shape))
    model.add(Conv2D(64, 5, 1, 'same'))
    model.add(Conv2D(64, 5, 1, 'same'))
    model.add(MaxPooling2D(2, 2, 'same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, 5, 1, 'same'))
    model.add(Conv2D(128, 5, 1, 'same'))
    model.add(MaxPooling2D(2, 2, 'same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def dnn(input_shape=None) -> Sequential:
    '''
    Initializes and compiles DNN.
    '''

    model = Sequential()
    if input_shape is not None:
        model.add(InputLayer(input_shape=input_shape))
    for _ in range(3):
        model.add(Dense(100, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(optimizer=RMSprop(learning_rate=3e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load(model: str, input_shape=None) -> Sequential:
    '''
    Initializes and compiles a model.
    '''

    assert model in MODELS

    if model == 'cnn':
        return cnn(input_shape=input_shape)

    if model == 'dnn':
        return dnn(input_shape=input_shape)
