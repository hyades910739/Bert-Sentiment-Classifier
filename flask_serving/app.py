import flask

from flask import Flask
from flask import render_template, url_for, request
import sys
import time
from pre_process import get_token
import json
from config import BERT_URL, SENTIMENT_URL
from predict import get_sentiment_predict, get_bert_embedding
from crawler import  get_raw_pushes_list_from_post, combine_user_push

app = Flask(__name__)


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        data = json.loads(data)
        lines = data['lines']
        if not lines:
            return {
                'message': 'no data found. should post a json with lines'
            }
        print(lines)
        token = dict(get_token(lines))
        embedding = get_bert_embedding(BERT_URL, token)
        output = get_sentiment_predict(SENTIMENT_URL, embedding)
        return {'output':output}
    return {
        'message': 'Post nothing.'
    }

@app.route('/predict_ptt',methods=['GET', 'POST'])
def predict_ptt():
    if request.method == 'POST':
        data = request.get_json()
        data = json.loads(data)
        url = data['url']
        posts = combine_user_push(get_raw_pushes_list_from_post(url))
        contents = [i[2] for i in posts]
        token = dict(get_token(contents))
        embedding = get_bert_embedding(BERT_URL, token)
        output = get_sentiment_predict(SENTIMENT_URL, embedding)
        line_and_score = [
            content + (score,) for content, score in zip(posts, output) 
        ]
        return {'output':line_and_score}


if __name__ == '__main__':
    app.run(
        host = '0.0.0.0',
        port = 8938,
        debug = True
    )
