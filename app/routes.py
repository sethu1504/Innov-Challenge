from app import app
from flask import request, jsonify
import app.utils as utils
import json


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
    print(dataset)
    return jsonify(dataset)


@app.route('/api/computeClusters', methods=['GET'])
def compute_clusters():
    k = int(request.args['clusterCount'])
    fields = request.args['fields'].split(',')
    dataset_id = int(request.args['id'])
    target_name = request.args['targetName']

    data = utils.get_clusters(dataset_id, k, fields, target_name)

    dataset = dict()
    dataset['data'] = data.to_json(orient='values')
    metadata = utils.read_metadata(dataset_id)
    metadata['metadata'].append(utils.get_cluster_field_metadata(target_name))
    dataset['metadata'] = metadata
    return jsonify(dataset)
