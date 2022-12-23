import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC

import datasets
import models
import submissions
import spec_augment

EPOCHS = 100
BATCH_SIZE = 32

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
    (_, x_train, y_train), _ = datasets.load(dataset, hot_one_encode=False, split=False)
    datagen = ImageDataGenerator(preprocessing_function=spec_augment.spec_augment)
    datagen.fit(x_train)

    # Define the SVC
    svc = SVC()

    # Train the SVC
    print(f'Training SVC…')
    svc_time = time.time()
    datagenerator = datagen.flow(x_train, y_train, batch_size=x_train.shape[0])
    x_batch, y_batch = datagenerator.next()
    x_batch = np.concatenate((x_batch, x_train), axis=0)
    y_batch = np.concatenate((y_batch, y_train), axis=0)
    svc.fit(x_batch.reshape(x_batch.shape[0], -1), y_batch)
    svc_time = time.time() - svc_time
    print(f'SVC trained in {svc_time:.2f}s.')

    # Load the new data
    (ids_train, x_train, y_train), (ids_val, x_val, y_val), (ids_test, x_test) = datasets.load(dataset, hot_one_encode=True, split=True)
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_val = x_val.reshape((x_val.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    # Update the data
    print(f'Updating data…')
    update_time = time.time()
    x_train = svc.predict(x_train)
    x_val = svc.predict(x_val)
    x_test = svc.predict(x_test)
    update_time = time.time() - update_time
    print(f'Data updated in {update_time:.2f}s.')

    # Define the DNN
    input_shape = (x_train.shape[1:])
    dnn = models.classifier(input_shape)

    # Train the DNN
    print(f'Training DNN…')
    dnn_time = time.time()
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
    hist = dnn.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_val, y_val), shuffle=True, callbacks=[early_stopping])
    dnn_time = time.time() - dnn_time
    print(f'DNN trained in {dnn_time:.2f}s.')

    # Evaluate the model
    print(f'Evaluating…')
    loss, accuracy = dnn.evaluate(x_val, y_val)
    total_time = svc_time + dnn_time
    print(f'Evaluation: {accuracy*100:.2f}%.')

    # Save and export summary
    Path(f'summaries/{dataset}/').mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(data=[[loss, accuracy, total_time]], columns=['Loss', 'Accuracy', 'Training Time'], dtype=str)
    summary.to_csv(f'summaries/{dataset}/{dataset}_{name}_summary.csv', index=False)

    # Create plots folder if necessary
    Path(f'plots/{dataset}/{name}').mkdir(parents=True, exist_ok=True)

    # Plot the loss
    plt.plot(hist.history['loss'], label='Training')
    plt.plot(hist.history['val_loss'], label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epoch')
    plt.savefig(f'plots/{dataset}/{name}/{dataset}_{name}_loss.png')
    plt.clf()

    # Plot the accuracy
    plt.plot(hist.history['accuracy'], label='Training')
    plt.plot(hist.history['val_accuracy'], label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epoch')
    plt.savefig(f'plots/{dataset}/{name}/{dataset}_{name}_accuracy.png')
    plt.clf()

    # Make the predictions
    print(f'Predicting…')
    y_pred = dnn.predict(x_test, batch_size=BATCH_SIZE).argmax(1) + 1
    Path(f'submissions/{dataset}').mkdir(parents=True, exist_ok=True)
    submissions.export(f'submissions/{dataset}/{dataset}_{name}_submission.csv', ids_test, y_pred)
