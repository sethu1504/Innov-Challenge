from app import app
from flask import jsonify
import pandas as pd
import os

@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"


@app.route('/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
	SITE_ROOT = os.path.realpath(os.path.dirname('data/credit.csv'))
	file_url = os.path.join(SITE_ROOT, 'credit.csv')
	data = pd.read_csv(file_url)
	return jsonify({'taskID': task_id})
