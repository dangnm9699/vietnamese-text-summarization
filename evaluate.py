import glob
import nltk
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import pickle
from revolution_score import bert_score_compute, rouge_score_compute
from pyvi import ViTokenizer
import numpy as np
from summarizer import Summarizer
import numpy as np

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


def read_plaintext(dir: str):
    dir = DATA_DIR + dir
    file = open(dir, 'r')
    plaintext = process(file.read())
    file.close()
    return plaintext


def read_manual_summary(dir: str):
    dir = MANUAL_DIR + dir
    file = open(dir, 'r')
    plaintext = process(file.read())
    file.close()
    return plaintext


def get_data():
    data_dir_list = []
    for dir in glob.glob('./data/plaintext/*/*.txt'):
        data_dir_list.append(dir[17:])
    return data_dir_list


def process(para: str):
    processed = ''
    for line in para.splitlines():
        line = line.strip()
        if line != '':
            if line[-1] != '.':
                line = line + '. '
            else:
                line = line + ' '
        processed += line
    return processed.strip()


if __name__ == '__main__':
    rouge_bert = []
    rouge_w2v = []
    # bert_bert = []
    # bert_w2v = []
    data_dir_list = get_data()
    for dir in data_dir_list:
        print(dir)
        manual_summary = read_manual_summary(dir)
        plaintext = plaintext = read_plaintext(dir)

        sentences = nltk.sent_tokenize(manual_summary)
        n_clusters = len(sentences)
        print(n_clusters)
        if n_clusters < 2:
            continue
        summary = ""
        try:
            summary = ''.join(model(
                body=plaintext,
                ratio=float(n_clusters),
                min_length=30,
                use_first=False
            ))
            summary = summary.replace('_', ' ')
            # p_, r_, f1_ = bert_score_compute(summary, manual_summary, 'vi')
            # bert_bert.append([p_, r_, f1_])
            p, r, f1 = rouge_score_compute(summary, manual_summary, '2')
            rouge_bert.append([p, r, f1])
        except AssertionError:
            pass
        except ValueError:
            pass

        summary = ""
        try:
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
            closest, _ = pairwise_distances_argmin_min(
                kmeans.cluster_centers_, X)
            ordering = sorted(range(n_clusters), key=lambda k: avg[k])
            summary = ' '.join([sentences[closest[idx]]
                                for idx in ordering])
        except ValueError:
            pass
        if summary != "":
            # try:
            #     p_, r_, f1_ = bert_score_compute(summary, manual_summary, 'vi')
            #     bert_w2v.append([p_, r_, f1_])
            #     print(bert_w2v)
            # except AssertionError:
            #     pass
            p, r, f1 = rouge_score_compute(summary, manual_summary, '2')
            rouge_w2v.append([p, r, f1])

    # print(bert_w2v)
    rouge_w2v = np.array(rouge_w2v)
    rouge_bert = np.array(rouge_bert)
    # bert_w2v = np.array(bert_w2v)
    # rouge_w2v = np.array(rouge_w2v)

    print(np.mean(rouge_w2v, axis=0))
    print(np.mean(rouge_bert, axis=0))
    # print(np.mean(bert_w2v, axis=0))
    # print(np.mean(rouge_w2v, axis=0))
