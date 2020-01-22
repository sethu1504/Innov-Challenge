import pandas as pd
import os
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

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


def compute_cluster_stats(data, cluster_field_name, cluster_fields):
    stats = dict()
    clusters = data[cluster_field_name].unique()

    for cluster in clusters:
        cluster_data = data[data[cluster_field_name] == cluster]
        cluster_data_stats = cluster_data.describe()
        cluster_stat = dict()
        cluster_stat['count'] = cluster_data.shape[0]
        fields = dict()
        for field in cluster_fields:
            field_stats = dict()
            print(cluster_data[field].dtype)
            field_stats['mean'] = cluster_data_stats[field]['mean']
            field_stats['min'] = cluster_data_stats[field]['min']
            field_stats['max'] = cluster_data_stats[field]['max']
            fields[field] = field_stats

        cluster_stat['fields'] = fields
        stats[cluster] = cluster_stat

    return stats


def get_mapping_points(data, cluster_fields):
    cluster_data = data[cluster_fields]
    distance = 1 - cosine_similarity(np.asarray(cluster_data))

    pca = PCA(2)
    pca.fit(distance)
    pca_data = pca.transform(distance)
    return pca_data
