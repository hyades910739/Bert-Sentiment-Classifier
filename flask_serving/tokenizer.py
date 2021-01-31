import tensorflow as tf
from transformers import AutoTokenizer, TFBertModel
chinese_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")