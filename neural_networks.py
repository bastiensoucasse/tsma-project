import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import datasets
import models
import submissions

EPOCHS = 20
BATCH_SIZE = 32

if __name__ == '__main__':
    # Check the usage.
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <model>.')
        print(f'    <dataset>: {datasets.DATASETS}.')
        print(f'    <model>: {models.MODELS}.')
        exit(-1)

    # Retrieve the dataset name.
    dataset_name = sys.argv[1].casefold()
    if dataset_name not in datasets.DATASETS:
        print(f'Dataset \'{dataset_name}\' unknown.')
        print(f'Available datasets: {datasets.DATASETS}.')
        exit(-1)

    # Retrieve the model name.
    model_name = sys.argv[2].casefold()
    if model_name not in models.MODELS:
        print(f'Model \'{model_name}\' unknown.')
        print(f'Available models: {models.MODELS}.')
        exit(-1)

    # Load the data.
    (ids_train, x_train, y_train), (ids_val, x_val, y_val), (ids_test, x_test) = datasets.load(dataset_name, hot_one_encode=True, split=True)

    # Define the input shape.
    if len(x_train.shape) <= 2:
        print(f'Dataset \'{dataset_name}\' incompatible.')
        print(f'The data must not be 1D.')
        exit(-1)
    input_shape = (x_train.shape[1], x_train.shape[2], 1)

    # Summarize the data.
    print(f'Train data shape: {x_train.shape} and {y_train.shape}.')
    print(f'Validation data shape: {x_val.shape} and {y_val.shape}.')
    print(f'Test data shape: {x_test.shape}.')

    # Define the model.
    model = models.load(model_name, input_shape=input_shape)
    print(f'Model: {model_name}.')

    # Train the model.
    start_time = time.time()
    hist = model.fit(x=x_train, y=y_train, batch_size=32, epochs=EPOCHS, validation_data=(x_val, y_val))
    training_time = time.time() - start_time

    # Evaluate the model.
    loss, accuracy = model.evaluate(x_val, y_val)

    # Print the model summary.
    print(f'Summary:\n    - Loss: {loss:.4f}\n    - Accuracy: {accuracy:.4f}\n    - Training Time: {training_time:.2f}s')

    # Save and export summary.
    Path(f'summaries/{dataset_name}/').mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(data=[[loss, accuracy, training_time]], columns=['Loss', 'Accuracy', 'Training Time'], dtype=str)
    summary.to_csv(f'summaries/{dataset_name}/{dataset_name}_{model_name}_summary.csv', index=False)

    # Create plots folder if necessary.
    Path(f'plots/{dataset_name}/{model_name}/').mkdir(parents=True, exist_ok=True)

    # Plot the loss.
    plt.plot(hist.history['loss'], label='Training')
    plt.plot(hist.history['val_loss'], label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epoch')
    plt.savefig(f'plots/{dataset_name}/{model_name}/{dataset_name}_{model_name}_loss.png')
    plt.clf()

    # Plot the accuracy.
    plt.plot(hist.history['accuracy'], label='Training')
    plt.plot(hist.history['val_accuracy'], label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epoch')
    plt.savefig(f'plots/{dataset_name}/{model_name}/{dataset_name}_{model_name}_accuracy.png')
    plt.clf()

    # Make the predictions.
    y_pred = model.predict(x_test, batch_size=BATCH_SIZE).argmax(1) + 1

    # Save and export predictions.
    Path(f'submissions/{dataset_name}/').mkdir(parents=True, exist_ok=True)
    submissions.export(f'submissions/{dataset_name}/{dataset_name}_{model_name}_submission.csv', ids_test, y_pred)
