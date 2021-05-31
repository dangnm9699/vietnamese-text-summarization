from flask import Flask, request
from flask import json
from flask.json import jsonify
from flask.templating import render_template
import nltk
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import pickle
from revolution_score import rouge_score_compute
from pyvi import ViTokenizer
import numpy as np
from summarizer import Summarizer

app = Flask(__name__)

DATA_DIR = './data/plaintext/'
MANUAL_DIR = './data/manual_summary/'

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


@app.route('/score', methods=['GET'])
def score_get():
    return render_template('score.html')


@app.route('/score', methods=['POST'])
def score_post():
    selected_model = request.args.get('model')
    if selected_model == 'bert':
        request_data = request.json
        ratio = float(request_data["ratio"])
        min_length = int(request_data["min_length"])
        plaintext_dir = DATA_DIR + str(request_data["path"])
        manual_summary_dir = MANUAL_DIR + str(request_data["path"])
        file = open(plaintext_dir, 'r')
        plaintext = file.read()
        bert_summary = ""
        file.close()
        for line in plaintext.splitlines():
            line = line.strip()
            if line != '' and line[-1:] != '.':
                line = line + '.'
            bert_summary += line.strip()
        file = open(manual_summary_dir, 'r')
        manual_summary = file.read()
        file.close()
        bert_summary = ''.join(model(
            body=bert_summary,
            ratio=ratio,
            min_length=0
        ))
        bert_summary = bert_summary.replace('_', ' ')
        p, r, f1 = rouge_score_compute(bert_summary, manual_summary, 'l')
        resp = {
            "model-summarized": bert_summary,
            "manual-summarized": manual_summary,
            "paragraph": plaintext,
            "p": p,
            "r": r,
            "f1": f1
        }
        return jsonify(resp)
    else:
        request_data = request.json
        plaintext_dir = DATA_DIR + str(request_data["path"])
        manual_summary_dir = MANUAL_DIR + str(request_data["path"])
        n_clusters = int(request_data["n_clusters"])
        # Read body
        file = open(plaintext_dir, 'r')
        plaintext = file.read()
        file.close()
        file = open(manual_summary_dir, 'r')
        manual_summary = file.read()
        file.close()
        sentences = nltk.sent_tokenize(plaintext)
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
        p, r, f1 = rouge_score_compute(summary, manual_summary, 'l')
        return jsonify({
            "model-summarized": ''.join(summary),
            "manual-summarized": manual_summary,
            "paragraph": plaintext,
            "p": p,
            "r": r,
            "f1": f1
        })


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
        line = line.strip()
        if line != '' and line[-1:] != '.':
            line = line + '.'
        paragraph += line.strip()

    result = ''.join(model(
        paragraph,
        ratio,
        min_length=0
    ))
    result = result.replace('_', ' ')
    resp = {
        "summarized": result
    }
    return jsonify(resp)
