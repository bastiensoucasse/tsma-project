import sys
from pathlib import Path

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from keras.preprocessing.image import ImageDataGenerator

import datasets
import submissions

MODELS = ['catboost', 'lightgbm', 'xgboost', 'svc']


def usage(message=None, quit=True):
    '''
    Prints a possible message and the program usage, and quits if needed.
    '''

    if message is not None:
        print(message)

    print(f'Usage: {sys.argv[0]} <dataset> <model> [\'data_augmentation\'].')
    print(f'    <dataset>: {datasets.DATASETS}.')
    print(f'    <model>: {MODELS}.')

    if quit:
        exit(-1)


if __name__ == '__main__':
    # Check the usage.
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        usage()

    # Retrieve the dataset name.
    dataset_name = sys.argv[1].casefold()
    if dataset_name not in datasets.DATASETS:
        usage(message=f'Dataset \'{dataset_name}\' unknown.')

    # Retrieve the model name.
    model_name = sys.argv[2].casefold()
    if model_name not in MODELS:
        usage(message=f'Model \'{model_name}\' unknown.')

    # Retrieve the data augmentation state.
    if len(sys.argv) == 4 and sys.argv[3].casefold() != 'data_augmentation':
        usage()
    data_augmentation = True if len(sys.argv) == 4 and sys.argv[3].casefold() == 'data_augmentation' else False

    # Load the data.
    (ids_train, x_train, y_train), (ids_test, x_test) = datasets.load(dataset_name, hot_one_encode=False, split=False)
    if model_name == 'xgboost':
        y_train -= 1

    # Reshape the data as 1D.
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    # Summarize the data.
    print(f'Dataset: {dataset_name}.')
    print(f'Train data shape: {x_train.shape} and {y_train.shape}.')
    print(f'Test data shape: {x_test.shape}.')

    # Define the model.
    if model_name == 'catboost':
        model = CatBoostClassifier(loss_function='MultiClass')
    if model_name == 'lightgbm':
        model = LGBMClassifier()
    if model_name == 'xgboost':
        model = XGBClassifier(use_label_encoder=False)
    if model_name == 'svc':
        model = SVC()
    print(f'Model: {model_name}.')

    # Define the data augmentation.
    if data_augmentation:
        print(f'Data augmentation activated.')
        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        datagen.fit(x_train)
        dataset_name += '_augmented'

    # Train the model.
    model.fit(datagen.flow(x_train, y_train, batch_size=1))

    # Make the predictions.
    y_pred = model.predict(x_test)
    if model_name == 'xgboost':
        y_pred += 1
    if model_name == 'catboost':
        y_pred = y_pred.reshape((len(y_pred)))

    # Save and export predictions.
    Path(f'submissions/{dataset_name}/').mkdir(parents=True, exist_ok=True)
    submissions.export(f'submissions/{dataset_name}/{dataset_name}_{model_name}_submission.csv', ids_test, y_pred)
