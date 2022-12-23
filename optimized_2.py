import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.svm import SVC

import datasets
import models
import submissions

EPOCHS = 100
BATCH_SIZE = 16

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


if __name__ == '__main__':
    # Check usage
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <name> <dataset>')

    # Retrieve the parameters
    name = sys.argv[1].casefold()
    dataset = sys.argv[2].casefold()

    # Load the data
    (ids_train, x_train, y_train), (ids_val, x_val, y_val), (ids_test, x_test) = datasets.load(dataset, hot_one_encode=True, split=True)

    # Define the CNN
    input_shape = (x_train.shape[1:])
    cnn = models.cnn(input_shape)

    # Train the CNN
    print(f'Training CNN…')
    cnn_time = time.time()
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
    hist = cnn.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_val, y_val), shuffle=True, callbacks=[early_stopping])
    cnn_time = time.time() - cnn_time
    print(f'DNN trained in {cnn_time:.2f}s.')

    # Evaluate the model
    print(f'Evaluating…')
    loss, accuracy = cnn.evaluate(x_val, y_val)
    print(f'Evaluation: {accuracy*100:.2f}%.')

    # Merge training and validation data
    x_train = np.concatenate((x_train, x_val), axis=0)
    y_train = np.concatenate((y_train, y_val), axis=0).argmax(1) + 1

    # Update the data
    x_train = cnn.predict(x_train, batch_size=BATCH_SIZE)
    x_test = cnn.predict(x_test, batch_size=BATCH_SIZE)

    # Reshape the data
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    # Save the data
    train = pd.DataFrame({"x": x_train, "y": y_train})
    train.to_csv("data/vggish_cnn_train.csv")
    test = pd.DataFrame({"x": x_test})
    test.to_csv("data/vggish_cnn_test.csv")

    # Define the SVC
    # svc = SVC()

    # Train the SVC
    # print(f'Training SVC…')
    # svc_time = time.time()
    # svc.fit(x_train, y_train)
    # svc_time = time.time() - svc_time
    # print(f'SVC trained in {svc_time:.2f}s.')

    # Make the predictions
    # print(f'Predicting…')
    # y_pred = svc.predict(x_test)
    # Path(f'submissions/{dataset}').mkdir(parents=True, exist_ok=True)
    # submissions.export(f'submissions/{dataset}/{dataset}_{name}_submission.csv', ids_test, y_pred)
