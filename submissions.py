import csv

import pandas as pd

TRACK_IDS = pd.read_csv('data/test.csv')['track_id'].to_numpy()


def _save(file_name, ids, y):
    file = open(file_name, 'w')
    writer = csv.writer(file, delimiter=',')

    header = ['track_id', 'genre_id']
    writer.writerow(header)

    for i in range(len(ids)):
        row = [ids[i], y[i]]
        writer.writerow(row)

    file.close()


def _fix(file_name):
    data = pd.read_csv(file_name)
    track_ids = data['track_id'].to_numpy()

    new = []
    for id in TRACK_IDS:
        if not id in track_ids:
            new += [[id, 7]]
    new = pd.DataFrame(new, columns=['track_id', 'genre_id'])
    data = pd.concat([data, new], ignore_index=True)

    data.to_csv(file_name, index=False)


def export(file_name: str, ids, y):
    '''
    Export a submission CSV file.
    '''

    _save(file_name, ids, y)
    _fix(file_name)
