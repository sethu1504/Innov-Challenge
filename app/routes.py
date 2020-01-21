from app import app
from flask import request
import app.utils as utils
import json


@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"


@app.route('/api/getData', methods=['GET'])
def get_data():
    dataset_id = int(request.args['id'])
    data = utils.read_dataset(dataset_id)
    return json.dumps(data)
