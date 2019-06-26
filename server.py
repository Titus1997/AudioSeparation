# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS, cross_origin
from werkzeug import secure_filename
from os import path
from getdrum import get_drum
from getvocal import get_vocal
from config import RunConfig
import logging

app = Flask(__name__)
CORS(app)
'''
@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
'''

app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def index():
    return "Welcome to the music separation webserver!"


@app.route('/drums', methods = ['GET', 'POST'])
@cross_origin()
def upload_drums():
   if request.method == 'POST':
      f = request.files['file']
      filename = secure_filename(f.filename)
      f.save(path.join(RunConfig.DATA_ROOT, filename))
      
      get_drum(filename)
      
      return send_file(path.join(RunConfig.RESULT_PATH, filename), mimetype="audio/wav", as_attachment=True)
	
@app.route('/vocals', methods = ['GET', 'POST'])
@cross_origin()
def upload_vocals():
   if request.method == 'POST':
      f = request.files['file']
      filename = secure_filename(f.filename)
      f.save(path.join(RunConfig.DATA_ROOT, filename))
      
      get_vocal(filename)
      
      return send_file(path.join(RunConfig.RESULT_PATH, filename), mimetype="audio/wav", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
