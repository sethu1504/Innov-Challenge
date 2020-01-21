import pandas as pd
import os
import json

DATA_FOLDER = os.path.realpath(os.path.dirname('data/'))


def read_dataset(data_id):
    dataset = dict()
    file_url = ''
    metadata_url = ''
    if data_id == 1:
        file_url = os.path.join(DATA_FOLDER, 'credit.csv')
        metadata_url = os.path.join(DATA_FOLDER, 'credit.json')
    elif data_id == 2:
        file_url = os.path.join(DATA_FOLDER, 'sports.csv')
        metadata_url = os.path.join(DATA_FOLDER, 'sports.json')

    data = pd.read_csv(file_url)
    with open(metadata_url) as json_file:
        metadata = json.loads(json_file.read())
        dataset['metadata'] = metadata['metadata']

    dataset['data'] = data.to_json(orient='values')
    return dataset