import pandas as pd
import os
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_FOLDER = os.path.realpath(os.path.dirname('data/'))

columns_1=['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT',
        'PAYMENTS', 'MINIMUM_PAYMENTS']


columns_2=['BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
         'CASH_ADVANCE_FREQUENCY', 'PRC_FULL_PAYMENT']


def read_dataset(data_id):
    file_url = ''
    if data_id == 1:
        file_url = os.path.join(DATA_FOLDER, 'credit.csv')
    elif data_id == 2:
        file_url = os.path.join(DATA_FOLDER, 'sports.csv')

    return pd.read_csv(file_url)


def convert_data_to_list(data):
    data = data.replace({pd.np.nan: None})
    data_dict_rows = data.to_dict(orient='records')
    data_list = []
    for row in data_dict_rows:
        data_list.append(list(row.values()))
    return data_list


def get_data(data_id):
    data = read_dataset(data_id)
    return convert_data_to_list(data)


def read_metadata(data_id):
    metadata_url = ''
    if data_id == 1:
        metadata_url = os.path.join(DATA_FOLDER, 'credit.json')
    elif data_id == 2:
        metadata_url = os.path.join(DATA_FOLDER, 'sports.json')
    with open(metadata_url) as json_file:
        metadata = json.loads(json_file.read())
        return metadata


def pre_process_data(data, fields):
    pre_processed_data = data[fields].copy()

    for field in fields:
        pre_processed_data.loc[(data[field].isnull() == True), field] = data[field].mean()

    #     if field in columns_1:
    #         pre_processed_data[col_name] = 0
    #         pre_processed_data.loc[((pre_processed_data[field] > 0) & (pre_processed_data[field] <= 500)), col_name] = 1
    #         pre_processed_data.loc[((pre_processed_data[field] > 500) & (pre_processed_data[field] <= 1000)), col_name] = 2
    #         pre_processed_data.loc[((pre_processed_data[field] > 1000) & (pre_processed_data[field] <= 3000)), col_name] = 3
    #         pre_processed_data.loc[((pre_processed_data[field] > 3000) & (pre_processed_data[field] <= 5000)), col_name] = 4
    #         pre_processed_data.loc[((pre_processed_data[field] > 5000) & (pre_processed_data[field] <= 10000)), col_name] = 5
    #         pre_processed_data.loc[(pre_processed_data[field] > 10000), col_name] = 6
    #
    #     elif field in columns_2:
    #         pre_processed_data[col_name] = 0
    #         pre_processed_data.loc[((pre_processed_data[field] > 0) & (pre_processed_data[field] <= 0.1)), col_name] = 1
    #         pre_processed_data.loc[((pre_processed_data[field] > 0.1) & (pre_processed_data[field] <= 0.2)), col_name] = 2
    #         pre_processed_data.loc[((pre_processed_data[field] > 0.2) & (pre_processed_data[field] <= 0.3)), col_name] = 3
    #         pre_processed_data.loc[((pre_processed_data[field] > 0.3) & (pre_processed_data[field] <= 0.4)), col_name] = 4
    #         pre_processed_data.loc[((pre_processed_data[field] > 0.4) & (pre_processed_data[field] <= 0.5)), col_name] = 5
    #         pre_processed_data.loc[((pre_processed_data[field] > 0.5) & (pre_processed_data[field] <= 0.6)), col_name] = 6
    #         pre_processed_data.loc[((pre_processed_data[field] > 0.6) & (pre_processed_data[field] <= 0.7)), col_name] = 7
    #         pre_processed_data.loc[((pre_processed_data[field] > 0.7) & (pre_processed_data[field] <= 0.8)), col_name] = 8
    #         pre_processed_data.loc[((pre_processed_data[field] > 0.8) & (pre_processed_data[field] <= 0.9)), col_name] = 9
    #         pre_processed_data.loc[((pre_processed_data[field] > 0.9) & (pre_processed_data[field] <= 1.0)), col_name] = 10
    #
    #     else:
    #         pre_processed_data[col_name] = 0
    #         pre_processed_data.loc[((pre_processed_data[field] > 0) & (pre_processed_data[field] <= 5)), col_name] = 1
    #         pre_processed_data.loc[((pre_processed_data[field] > 5) & (pre_processed_data[field] <= 10)), col_name] = 2
    #         pre_processed_data.loc[((pre_processed_data[field] > 10) & (pre_processed_data[field] <= 15)), col_name] = 3
    #         pre_processed_data.loc[((pre_processed_data[field] > 15) & (pre_processed_data[field] <= 20)), col_name] = 4
    #         pre_processed_data.loc[((pre_processed_data[field] > 20) & (pre_processed_data[field] <= 30)), col_name] = 5
    #         pre_processed_data.loc[((pre_processed_data[field] > 30) & (pre_processed_data[field] <= 50)), col_name] = 6
    #         pre_processed_data.loc[((pre_processed_data[field] > 50) & (pre_processed_data[field] <= 100)), col_name] = 7
    #         pre_processed_data.loc[(pre_processed_data[field] > 100), col_name] = 8
    #
    # pre_processed_data = pre_processed_data.drop(fields, axis=1)
    scale = StandardScaler()
    pre_processed_data[fields] = scale.fit_transform(pre_processed_data[fields])
    return pre_processed_data


def get_clusters(data_id, k, cluster_fields, target_name, detect_anomaly):
    data = read_dataset(data_id)
    processed_data = pre_process_data(data, cluster_fields)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(np.asarray(processed_data))

    if detect_anomaly:
        processed_data['intermediate_cluster'] = kmeans.labels_
        processed_data['dist'] = 0

        for i in range(0, k):
            processed_data.loc[(processed_data['intermediate_cluster'] == i), 'dist'] = \
                processed_data[cluster_fields].sub(np.array(kmeans.cluster_centers_[i])).pow(2).sum(1).pow(0.5)

        processed_data[target_name] = processed_data['intermediate_cluster']
        for i in range(0, k):
            cluster_mean = processed_data[processed_data['intermediate_cluster'] == i]['dist'].mean()
            cluster_std = processed_data[processed_data['intermediate_cluster'] == i]['dist'].std()

            limit = cluster_mean + (3 * cluster_std)

            processed_data.loc[(processed_data['intermediate_cluster'] == i) & (processed_data['dist'] > limit), target_name] = -1

        data[target_name] = processed_data[target_name]
        return data
    else:
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
        description = ''
        cluster_data = data[data[cluster_field_name] == cluster]
        cluster_data_stats = cluster_data.describe()
        cluster_stat = dict()
        cluster_stat['count'] = cluster_data.shape[0]
        fields = dict()
        for field in cluster_fields:
            max_field_value = max(data[field])
            min_field_value = min(data[field])
            field_stats = dict()
            if cluster_data[field].dtype == float:
                field_stats['mean'] = cluster_data_stats[field]['mean']
                field_stats['min'] = cluster_data_stats[field]['min']
                field_stats['max'] = cluster_data_stats[field]['max']
                description += _get_prefix(max_field_value, min_field_value, field_stats['max'], field_stats['min'])
            else:
                unique_dimensions = cluster_data[field].unique()
                for dimension in unique_dimensions:
                    field_stats[dimension] = cluster_data[cluster_data[field] == dimension].shape[0]
                description += 'Unique'
            field_stats['bins'] = _get_field_bins(field, cluster_data[field])
            fields[field] = field_stats
            description += ' ' + field + ', '

        cluster_stat['fields'] = fields
        cluster_stat['description'] = description[:-2]
        stats['Cluster' + str(cluster)] = cluster_stat

    return stats


def get_mapping_points(data, cluster_fields, target_name):
    processed_data = pre_process_data(data, cluster_fields)
    distance = 1 - cosine_similarity(np.asarray(processed_data))

    pca = PCA(2)
    pca.fit(distance)
    pca_data = pca.transform(distance)

    mapping_points = []
    for i in range(0, pca_data.shape[0]):
        record = list(pca_data[i])
        record.append(str(data.iloc[i][target_name]))
        mapping_points.append(record)
    return mapping_points


def _get_prefix(overall_max, overall_min, current_max, current_min):
    if current_max == overall_max:
        return 'High'
    elif current_min == overall_min:
        return 'Low'
    else:
        return 'Average'


def _get_field_bins(field, field_data):
    bins = dict()
    bin_count = _get_bin_count(field)
    for count in range(bin_count):
        # initialize each bin to count 0
        bins[count + 1] = 0
    for data in field_data:
        key = int(data % bin_count) + 1
        bins[key] += 1
    return bins


def _get_bin_count(field):
    if field in ['BALANCE', 'PURCHASES', 'ONE_OFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS']:
        return 6
    elif field in ['PURCHASES_TAX', 'CASH_ADVANCE_TAX']:
        return 8
    elif field in ['TENURE']:
        return 12
    elif field in ['ONE_OFF_PURCHASES_FREQUENCY']:
        return 20
    else:
        return 10
