import pandas as pd
import os
import json
from sklearn.cluster import KMeans

DATA_FOLDER = os.path.realpath(os.path.dirname('data/'))


def read_dataset(data_id):
    file_url = ''
    if data_id == 1:
        file_url = os.path.join(DATA_FOLDER, 'credit.csv')
    elif data_id == 2:
        file_url = os.path.join(DATA_FOLDER, 'sports.csv')

    return pd.read_csv(file_url)


def read_metadata(data_id):
    metadata_url = ''
    if data_id == 1:
        metadata_url = os.path.join(DATA_FOLDER, 'credit.json')
    elif data_id == 2:
        metadata_url = os.path.join(DATA_FOLDER, 'sports.json')
    with open(metadata_url) as json_file:
        metadata = json.loads(json_file.read())
        return metadata


def get_clusters(data_id, k, fields, target_name):
    data = read_dataset(data_id)
    kmeans = KMeans(n_clusters=k).fit(data[fields].values)
    data[target_name] = kmeans.labels_
    return data


def get_cluster_field_metadata(field_name):
    field_props = dict()
    field_props['name'] = field_name
    field_props['fullyQualifiedName'] = field_name.replace(' ', '_')
    field_props['type'] = 'TEXT'
    field_props['label'] = field_name
    field_props['description'] = None
    field_props['length'] = None
    field_props['multiValue'] = False
    return field_props
