from app import app
from flask import request, jsonify
import app.utils as utils


@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"


@app.route('/api/getData', methods=['GET'])
def get_data():
    dataset_id = int(request.args['id'])

    dataset = dict()
    dataset['data'] = utils.get_data(dataset_id)
    dataset['metadata'] = utils.read_metadata(dataset_id)
    return jsonify(dataset)


@app.route('/api/computeClusters', methods=['GET'])
def compute_clusters():
    k = int(request.args['clusterCount'])
    fields = request.args['fields'].split(',')
    dataset_id = int(request.args['id'])
    target_name = request.args['targetName']
    detect_anomaly = int(request.args['detectAnomaly'])
    cluster_names = []
    if 'clusterNames' in request.args:
        cluster_names = request.args['clusterNames'].split(',')

    data = utils.get_clusters(dataset_id, k, fields, target_name, detect_anomaly, cluster_names)

    dataset = dict()
    dataset['data'] = utils.convert_data_to_list(data)
    metadata = utils.read_metadata(dataset_id)
    metadata.append(utils.get_cluster_field_metadata(target_name))
    dataset['metadata'] = metadata
    dataset['mappingPoints'] = utils.get_mapping_points(data, fields, target_name)
    dataset['stats'] = utils.compute_cluster_stats(data, target_name, fields)
    return jsonify(dataset)
