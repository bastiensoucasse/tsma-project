import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import datasets
import models
import submissions

EPOCHS = 100
BATCH_SIZE = 32

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


if __name__ == '__main__':
    # Check usage
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <name>')

    # Retrieve the parameters
    name = sys.argv[1].casefold()
    dataset = 'opt'

    # Load the data
    (ids_train, x_train, y_train), (ids_test, x_test) = datasets.opt()

    # Define the SVC
    svc = SVC()

    # Train the SVC
    print(f'Training SVC…')
    svc_time = time.time()
    svc.fit(x_train, y_train)
    svc_time = time.time() - svc_time
    print(f'SVC trained in {svc_time:.2f}s.')

    # Update the data
    print(f'Updating data…')
    update_time = time.time()
    x_train = svc.predict(x_train)
    x_test = svc.predict(x_test)
    ids_train, ids_val, x_train, x_val, y_train, y_val = train_test_split(ids_train, x_train, y_train, test_size=0.2)
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
