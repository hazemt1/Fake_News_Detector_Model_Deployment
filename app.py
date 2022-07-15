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
from arabert.preprocess import ArabertPreprocessor
from langdetect import detect, DetectorFactory
import jdk


path_to_java_home = jdk.install('15', jre=True)
DetectorFactory.seed = 0

nltk.download('stopwords')
nltk.download('punkt')

# *******************************************************************************************************************
arabert_model_name = "aubmindlab/bert-base-arabertv2"
arabert_prep = ArabertPreprocessor(model_name=arabert_model_name,)
stops = set(stopwords.words("arabic"))
port_stem = SnowballStemmer('english')

def remove_stop_words(text):
    zen = TextBlob(text)
    words = zen.words
    return " ".join([w for w in words if not w in stopwords.words('english') and len(w) >= 2])


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
    # text = re.sub("\d+", " ", text)
    text = re.sub(r'\\u[A-Za-z0-9\\]+', ' ', text)
    text = re.sub('\n+', ' ', text)
    zen = TextBlob(text)
    words = zen.words
    text = [port_stem.stem(word) for word in words]
    text = ' '.join(text)
    text = arabert_prep.preprocess(text)
    text = re.sub('(\w+\+)|(\+\w+)', '', text)
    text = re.sub('\s+', ' ', text)
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
global ar_tokenizer
global en_tokenizer
global ar_model
global ar_model_weights
global ar_model_json
global en_model
global en_model_weights
global en_model_json
v = 0
en_tokenizer = download_item('en_tokenizer')
en_model_weights = download_item('english_weights$1.pickle')
en_model_json = download_item('english_json.pickle')

ar_tokenizer = download_item('ar_tokenizer.pickle')
ar_model_weights = download_item('ar_weights_2.pickle')
ar_model_json = download_item('ar_json_2.pickle')

# en_model_json = pickle.load(open('english_json.pickle', 'rb'))
# en_model_weights = pickle.load(open('english_weights.pickle', 'rb'))
# en_tokenizer = pickle.load(open('en_tokenizer', 'rb'))

# ar_model_json = pickle.load(open('ar_json.pickle', 'rb'))
# ar_model_weights = pickle.load(open('ar_weights.pickle', 'rb'))
# ar_tokenizer = pickle.load(open('ar_tokenizer.pickle', 'rb'))

en_model = keras.models.model_from_json(en_model_json)
en_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['acc'])
en_model.set_weights(en_model_weights)

ar_model = keras.models.model_from_json(ar_model_json)
ar_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['acc'])
ar_model.set_weights(ar_model_weights)

app = Flask(__name__)


@app.route('/', methods=['POST'])
def predict():
    global v
    global ar_tokenizer
    global en_tokenizer
    global ar_model
    global en_model
    json = request.json
    news = json['news']
    lang = detect(news)
    news = np.array([news])
    news = pd.DataFrame(news, columns=['claim_s'])
    news.astype(str)
    news = news['claim_s'].apply(lambda x: clean_text(x))
    if lang == 'ar':
        news = ar_tokenizer.texts_to_sequences(news)
        news = pad_sequences(news, maxlen=1000)
        prediction = ar_model.predict(news)
    elif lang == 'en':
        news = en_tokenizer.texts_to_sequences(news)
        news = pad_sequences(news, maxlen=1000)
        prediction = en_model.predict(news)
    else:
        response = {
            "result": -1,
            "version": v
        }
        return jsonify(response)
    output = prediction[0][0]
    response = {
        "result": float(output),
        "version": v
    }
    return jsonify(response)


@app.route('/update', methods=['post'])
def update():
    global v
    global ar_tokenizer
    global en_tokenizer
    global ar_model
    global ar_model_weights
    global ar_model_json
    global en_model
    global en_model_weights
    global en_model_json

    en_tokenizer = download_item('en_tokenizer')
    en_model_weights = download_item('english_weights$1.pickle')
    en_model_json = download_item('english_json.pickle')

    ar_tokenizer = download_item('ar_tokenizer.pickle')
    ar_model_weights = download_item('ar_weights_2.pickle')
    ar_model_json = download_item('ar_json_2.pickle')

    en_model = keras.models.model_from_json(en_model_json)
    en_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['acc'])
    en_model.set_weights(en_model_weights)

    ar_model = keras.models.model_from_json(ar_model_json)
    ar_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['acc'])
    ar_model.set_weights(ar_model_weights)
    v += 1
    response = {
        "result": "success"
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(port=8000,)
