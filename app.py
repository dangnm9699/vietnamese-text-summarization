from flask import Flask, request
from flask import json
from flask.json import jsonify
from flask.templating import render_template
from summarizer import Summarizer

app = Flask(__name__)

model = Summarizer()


@app.route('/')
def index():
    return render_template('index.html')


'''
    Summarize API
    request body = {
        body: '',
        ratio: 0.4,
        min_length: 30
    }
'''


@app.route('/summarize', methods=['POST'])
def summarizer():
    data = request.json
    ratio = float(data["ratio"])
    min_length = int(data["min_length"])
    body = data["body"]
    result = ''.join(model(
        body,
        ratio,
        min_length
    ))
    resp = {
        "summarized": result
    }
    return jsonify(resp)
