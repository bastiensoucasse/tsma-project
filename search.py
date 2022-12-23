import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from keras.applications import ResNet50
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, InputLayer, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import RandomizedSearchCV

import datasets
import models
import submissions
import spec_augment

EPOCHS = 100

CLASSES = {'Classical': 1, 'Electronic': 2, 'Folk': 3, 'Hip-Hop': 4, 'World Music': 5, 'Experimental': 6, 'Pop': 7, 'Rock': 8}
NUM_CLASSES = len(CLASSES)

params = {
    'layer_groups': [1, 2, 3],
    'layers_in_group': [1, 2, 3],
    'learning_rate': [1e-2, 1e-3, 1e-4, 1e-5],
    'batch_size': [32, 64, 128]
}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


if __name__ == '__main__':
    # Load the data.
    (ids_train, x_train, y_train), (ids_val, x_val, y_val), (ids_test, x_test) = datasets.load('vggish', hot_one_encode=True, split=True)

    # Define the input shape.
    input_shape = (x_train.shape[1], x_train.shape[2], 1)


    def build_and_train_model(layer_groups: int, layers_in_group: int, learning_rate: float, batch_size: int):
        model = Sequential()
        model.add(InputLayer(input_shape=input_shape))
        model.add(Flatten())
        for i in range(layer_groups):
            for _ in range(layers_in_group):
                model.add(Dense((i + 1) * 1024, activation='relu'))
            model.add(Dropout(0.25))
        model.add(Dense(NUM_CLASSES, activation='softmax'))

        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=batch_size, epochs=50, verbose=0, validation_data=(x_val, y_val))
        _, accuracy = model.evaluate(x_val, y_val, return_dict=True)
        return accuracy['accuracy']


    # Search the best model.
    search = RandomizedSearchCV(estimator=None, param_distributions=params, scoring='accuracy', n_iter=30, n_jobs=-1, cv=3)
    search.fit(x_train, y_train)

    # Print the best hyperparameters and the corresponding validation accuracy
    print(search.best_params_)
    print(search.best_score_)

    # Train the model with the best hyperparameters and evaluate on the test set
    model = build_and_train_model(search.best_params_['learning_rate'], search.best_params_['batch_size'])
    test_loss, test_accuracy = model.evaluate(x_val, y_val, return_dict=True)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_accuracy)
    print(model)
