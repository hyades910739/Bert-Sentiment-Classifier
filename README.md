# Bert-Sentiment-Classifier

使用[huggingface](https://huggingface.co/transformers/)提供的bert模型，搭配flask與tf-serving，可以快速訓練並建立預測API。

## requirements:
* python3.6
* Docker 19.03.12
* tensorflow==2.4.0
* transformers==4.2.2
* pyquery==1.4.3
* Flask==1.1.2

## How to use.
1. 導出一個bert model(不用訓練)。
```
#python
# easy as F
from transformers import TFBertModel
bert_model = TFBertModel.from_pretrained("bert-base-chinese")
bert_model.save_pretrained("bert_chinese", saved_model=True)
```
2. 訓練(transfer learning)一個bert分類器，並save成pb格式。 (參考`transfer.py`)
3. 用Docker開bert與分類器的serving container ([教學](https://www.tensorflow.org/tfx/serving/docker))
```
docker run -t -p 8501:8501 -v 'MODELPATH:/models/bert' -e MODEL_NAME=bert tensorflow/serving &

docker run -t -p 8503:8501 -v 'MODELPATH:/models/sentiment_classifier' -e MODEL_NAME=sentiment_classifier tensorflow/serving &
```
4. 開flask
5. 預測
```
#python
# predict Lines
my_data = json.dumps({'lines': ['你好嗎', '我很好', '不想上班']})
r = requests.post('http://localhost:8938/predict', json = my_data)

# predict comments from ptt.
my_data = json.dumps({'url':'https://www.ptt.cc/bbs/Boy-Girl/M.1612021602.A.139.html'})
r = requests.post('http://localhost:8938/predict_ptt', json = my_data)
```

## notes:
* transfer的時候，如果沒有要訓練bert層，可以先用bert預測data的embedding，再用embedding訓練下游的分類器。可以大大的減少訓練時間(連bert一起訓練的話GPU應該會炸掉)
* 輸出成兩個模型，以後要換上游embedding或下游分類器都比較方便
