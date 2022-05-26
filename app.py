import numpy as np
import pandas as pd
import pickle
import h5py
import re
import nltk
import urllib
import ast
from flask import Flask, request, jsonify
from keras.models import load_model
from azure.storage.blob import BlobClient
from io import BytesIO
from textblob import TextBlob
from nltk.corpus import stopwords
# from keras.utils.data_utils import pad_sequences
from keras.preprocessing.sequence import pad_sequences
from nltk.stem import SnowballStemmer


nltk.download('stopwords')
nltk.download('punkt')


with urllib.request.urlopen('https://detect0rnews.blob.core.windows.net/newcontainer/tokenizer.pickle') as f:
    a = str(bytes(f.read()))
# *******************************************************************************************************************

stops = set(stopwords.words("arabic"))
port_stem = SnowballStemmer('english')


def remove_stop_words(text):
    zen = TextBlob(text)
    words = zen.words
    return " ".join([w for w in words if not w in stops and not w in stopwords.words('english') and len(w) >= 2])


def split_hashtag_to_words(tag):
    tag = tag.replace('#', '')
    tags = tag.split('_')
    if len(tags) > 1:
        return tags
    pattern = re.compile(r"[A-Z][a-z]+|\d+|[A-Z]+(?![a-z])")
    return pattern.findall(tag)


def clean_hashtag(text):
    words = text.split()
    text = list()
    for word in words:
        if is_hashtag(word):
            text.extend(extract_hashtag(word))
        else:
            text.append(word)
    return " ".join(text)


def is_hashtag(word):
    if word.startswith("#"):
        return True
    else:
        return False


def extract_hashtag(text):
    hash_list = ([re.sub(r"(\W+)$", "", i) for i in text.split() if i.startswith("#")])
    word_list = []
    for word in hash_list:
        word_list.extend(split_hashtag_to_words(word))
    return word_list


def clean_tweet(text):
    text = re.sub('#\d+K\d+', ' ', str(text))  # years like 2K19
    text = re.sub('http\S+\s*', ' ', str(text))  # remove URLs
    text = re.sub('RT|cc', ' ', str(text))  # remove RT and cc
    text = re.sub('@[^\s]+', ' ', str(text))
    text = clean_hashtag(str(text))
    return text


def clean_text(text):
    text = clean_tweet(text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)  # remove punctuation
    text = re.sub('\s+', ' ', text)
    text = text.lower()
    text = remove_stop_words(text)
    text = re.sub("\d+", " ", text)
    text = re.sub(r'\\u[A-Za-z0-9\\]+', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = [port_stem.stem(word) for word in text]
    text = ' '.join(text)

    return text


# *******************************************************************************************************************
con_str = 'DefaultEndpointsProtocol=https;AccountName=detect0rnews;AccountKey=+T/vwDH865hqfCeAZsSooIPtaLgH+fXwUbfMqT7t8i0dXjgEG1yvfIj83EKCwzVqCwxINo3yRtIz+AStID/rlg==;EndpointSuffix=core.windows.net'
blob_client = BlobClient.from_connection_string(con_str, blob_name='rnn_model.h5', container_name='newcontainer')
downloader = blob_client.download_blob(0)

with BytesIO() as f:
    downloader.readinto(f)
    with h5py.File(f, 'r') as h5file:
        model = load_model(h5file)
tokenizer = pickle.loads(ast.literal_eval(a))

app = Flask(__name__)


@app.route('/', methods=['POST'])
def predict():
    json = request.json
    news = json['news']
    news = np.array([news])
    news = pd.DataFrame(news, columns=['claim_s'])
    news = news['claim_s'].apply(lambda x: clean_text(x))
    news = tokenizer.texts_to_sequences(news)
    news = pad_sequences(news, maxlen=1000)
    prediction = model.predict(news)

    output = prediction[0][0]

    response = {
        "result": float(output)
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(port=8000)
