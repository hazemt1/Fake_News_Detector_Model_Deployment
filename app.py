import numpy as np
import pandas as pd
import pickle
import re
import nltk
from flask import Flask, request, jsonify
import keras
from azure.storage.blob import BlobClient
from textblob import TextBlob
from nltk.corpus import stopwords
from keras.utils.data_utils import pad_sequences
# from keras.preprocessing.sequence import pad_sequences
from nltk.stem import SnowballStemmer
from nltk import ISRIStemmer

nltk.download('stopwords')
nltk.download('punkt')

# *******************************************************************************************************************

stops = set(stopwords.words("arabic"))
port_stem = SnowballStemmer('english')
ArListem = ISRIStemmer()

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


def stem(text):
    zen = TextBlob(text)
    words = zen.words
    cleaned = list()
    for w in words:
        cleaned.append(ArListem.stem(w))
    return " ".join(cleaned)


def clean_text(text):
    text = clean_tweet(text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)  # remove punctuation
    text = re.sub('\s+', ' ', text)
    text = text.lower()
    text = remove_stop_words(text)
    text = re.sub("\d+", " ", text)
    text = re.sub(r'\\u[A-Za-z0-9\\]+', ' ', text)
    text = re.sub('\s+', ' ', text)
    zen = TextBlob(text)
    words = zen.words
    text = [port_stem.stem(word) for word in words]
    text = ' '.join(text)
    text = stem(text)

    return text


con_str = 'DefaultEndpointsProtocol=https;AccountName=detect0rnews;AccountKey=+T/vwDH865hqfCeAZsSooIPtaLgH+fXwUbfMqT7t8i0dXjgEG1yvfIj83EKCwzVqCwxINo3yRtIz+AStID/rlg==;EndpointSuffix=core.windows.net'


def download_item(item_name):
    blob_client = BlobClient.from_connection_string(con_str, blob_name=item_name, container_name='newcontainer')
    downloader = blob_client.download_blob(0)
    f = downloader.readall()
    item = pickle.loads(f)
    return item


# *******************************************************************************************************************
global v
global tokenizer
global model
global model_weights
global model_json
v = 0
tokenizer = download_item('tokenizer.pickle')

model_weights = download_item('model_weights.pkl')
model_json = download_item('model_json.pkl')

model = keras.models.model_from_json(model_json)
model.set_weights(model_weights)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['acc'])

app = Flask(__name__)


@app.route('/', methods=['POST'])
def predict():
    global v
    global tokenizer
    global model
    global model_weights
    global model_json
    json = request.json
    news = json['news']
    news = np.array([news])
    news = pd.DataFrame(news, columns=['claim_s'])
    news = news['claim_s'].apply(lambda x: clean_text(x))
    print(news)
    news = tokenizer.texts_to_sequences(news)
    news = pad_sequences(news, maxlen=1000)
    prediction = model.predict(news)

    output = prediction[0][0]
    response = {
        "result": float(output),
        "version": v
    }
    return jsonify(response)


@app.route('/update', methods=['post'])
def update():
    global tokenizer
    global v
    global model
    global model_weights
    global model_json
    tokenizer = download_item('tokenizer.pickle')

    model_weights = download_item('model_weights.pkl')
    model_json = download_item('model_json.pkl')

    model = keras.models.model_from_json(model_json)
    model.set_weights(model_weights)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['acc'])
    v += 1
    response = {
        "result": "success"
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(port=8000)
