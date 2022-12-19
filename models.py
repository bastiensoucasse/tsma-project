from keras.applications import ResNet50
from keras.layers import BatchNormalization, Conv2D, Dense, Flatten, InputLayer, MaxPooling2D
from keras.models import Sequential

MODELS = ['cnn', 'dnn', 'resnet']

CLASSES = {'Classical': 1, 'Electronic': 2, 'Folk': 3, 'Hip-Hop': 4, 'World Music': 5, 'Experimental': 6, 'Pop': 7, 'Rock': 8}
NUM_CLASSES = len(CLASSES)


def cnn(input_shape):
    '''
    Initializes and compiles CNN.
    '''

    model = Sequential()
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


def dnn(input_shape):
    '''
    Initializes and compiles DNN.
    '''

    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def resnet(input_shape):
    '''
    Initializes and compiles ResNet.
    '''

    model = ResNet50(weights=None, input_shape=input_shape, classes=NUM_CLASSES)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
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

    if model == 'resnet':
        return resnet(input_shape=input_shape)
