from flask import Flask, request
from flask import json
from flask.json import jsonify
from flask.templating import render_template
import nltk
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import pickle

from pyvi import ViTokenizer
import numpy as np
from summarizer import Summarizer

app = Flask(__name__)

print("Initialize Summarizer...")
model = Summarizer()
print("Done")

print("Downloading punkt...")
try:
    nltk.data.find('tokenizers/punkt')
    print('Existed')
except LookupError:
    nltk.download('punkt')
    print("Downloaded")

vocab = None
print("Loading vocab...")
try:
    vocab_file = open("vocab.pkl", "rb")
    vocab = pickle.load(vocab_file)
    vocab_file.close()
except Exception:
    w2v = KeyedVectors.load_word2vec_format('we_knn/wiki.vi.vec')
    vocab = w2v.key_to_index
    vocab_file = open("vocab.pkl", "wb")
    pickle.dump(vocab, vocab_file)
    vocab_file.close()
print("Done")


@app.route('/')
def index():
    return render_template('base.html')


@app.route('/knn', methods=['GET'])
def knn_get():
    return render_template('knn.html')


@app.route('/knn', methods=['POST'])
def knn_post():
    data = request.json
    body = str(data["body"])
    n_clusters = int(data["n_clusters"])
    sentences = nltk.sent_tokenize(body)
    X = []
    for sentence in sentences:
        sentence = ViTokenizer.tokenize(sentence)
        words = sentence.split(" ")
        sentence_vec = np.zeros((300))
        for word in words:
            if word in vocab:
                sentence_vec += vocab[word]
                break
        X.append(sentence_vec)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)

    avg = []
    for j in range(n_clusters):
        idx = np.where(kmeans.labels_ == j)[0]
        avg.append(np.mean(idx))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])
    summary = ' '.join([sentences[closest[idx]] for idx in ordering])
    return jsonify({"summarized": ''.join(summary)})


@app.route('/bert', methods=['GET'])
def bert_get():
    return render_template('bert.html')


'''
    Summarize API
    request body = {
        body: '',
        ratio: 0.4,
        min_length: 30
    }
'''


@app.route('/bert', methods=['POST'])
def bert_post():
    data = request.json
    ratio = float(data["ratio"])
    min_length = int(data["min_length"])
    body = str(data["body"])
    paragraph = ""
    for line in body.splitlines():
        paragraph += line.strip()

    result = ''.join(model(
        paragraph,
        ratio,
        min_length
    ))
    result = result.replace('_', ' ')
    resp = {
        "summarized": result
    }
    return jsonify(resp)
