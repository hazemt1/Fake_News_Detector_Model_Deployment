import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import pickle
import keras
from gensim.models import Word2Vec
import qalsadi.lemmatizer
import re
from textblob import TextBlob
from nltk.corpus import stopwords
from keras.utils.data_utils import pad_sequences
# from keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import SnowballStemmer

# *******************************************************************************************************************

stops = set(stopwords.words("arabic"))
lemmer = qalsadi.lemmatizer.Lemmatizer()
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
    try:
        text = lemmer.lemmatize_text(text)
    except:
        print('failed')
    text = [port_stem.stem(word) for word in text]
    text = ' '.join(text)

    return text


# *******************************************************************************************************************

app = Flask(__name__)
model = keras.models.load_model('rnn_model.h5')
tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))
w2v = Word2Vec.load("word2vec.model")


@app.route('/', methods=['POST'])
def predict():
    json = request.json
    news = json['news']
    news = np.array([news])
    news = pd.DataFrame(news, columns=['claim_s'])
    print(news)
    news = news['claim_s'].apply(lambda x: clean_text(x))
    news = tokenizer.texts_to_sequences(news)
    news = pad_sequences(news, maxlen=1000)
    prediction = model.predict(news)

    output = prediction[0][0]

    response = {
        "result": float(output)
    }
    print(output)
    return jsonify(response)


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(port=8000)
