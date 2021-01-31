import requests
import json

def get_bert_embedding(post_url, data): # emb: List
    'data: dict of model input.'
    # if 'inputs' not in data:
    #     raise Exception("Failed calling bert embedding: only support 'inputs' as key for data.")
    data = {'inputs': data}
    val = requests.post(post_url, data=json.dumps(data))
    if val.status_code != 200:
        raise Exception(
            'Failed calling bert embedding: status_code:{}. {}'.format(val.status_code, val.json())
        )
    emb = val.json()['outputs']['pooler_output']
    return emb

def get_sentiment_predict(post_url, data): #-> List[List]
    'data: List[List]. the bert embedding list.'
    sample = {
        'signature_name': 'call',
        'inputs':{'input':data}
    }
    val = requests.post(post_url, data=json.dumps(sample))    
    if val.status_code != 200:
        raise Exception(
            'Failed calling sentiment classifier: status_code:{}. {}'.format(val.status_code, val.json())
        )
    return val.json()['outputs']