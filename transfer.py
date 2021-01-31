#util.py
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFBertModel

import tensorflow as tf
from tqdm import tqdm


from model import DownstreamClassifier

hyper_params = {
    'lr': 0.001,
    'batch_size': 64,
    'epochs': 20,
    'hidden_dims': [768, 1200, 768],
}

def get_dataset(x, y):    
    train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=689, test_size=0.2)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    return train_dataset, test_dataset

def train():
    labels = np.load('labels.npy')
    labels = np.array([[1,0] if i else [0,1] for i in labels]) #two 2-dim label
    lines_embedding = np.load('lines_embedding.npy')
    train_dataset, test_dataset = get_dataset(lines_embedding, labels)
    model = DownstreamClassifier([768, 1200, 768], 0.3, bn=True, output_unit=2)
    
    criteon = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyper_params['lr'])
    train_auc = tf.keras.metrics.AUC()
    val_auc = tf.keras.metrics.AUC()
    train_bc = tf.keras.metrics.BinaryCrossentropy()
    val_bc = tf.keras.metrics.BinaryCrossentropy()

    for epoch in range(hyper_params['epochs']):
        train_auc.reset_states()
        val_auc.reset_states()
        train_bc.reset_states()
        val_bc.reset_states()

        for x, y in tqdm(train_dataset.shuffle(1000).batch(hyper_params['batch_size'])): 
            with tf.GradientTape() as tape:
                logits = model.train(x)
                loss = criteon(y, logits)  
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_auc.update_state(y, logits)
            train_bc.update_state(y, logits)


        for x, y in test_dataset.batch(256): 
            logits = model(x)
            loss = criteon(y, logits)
            val_auc.update_state(y, logits)
            val_bc.update_state(y, logits)        

        print('+ epoch : ',epoch)
        print('    * train loss:', train_bc.result().numpy())
        print('    * val loss:', val_bc.result().numpy())
        print('    * train AUC:', train_auc.result().numpy())
        print('    * val AUC:', val_auc.result().numpy())    
    tf.saved_model.save(model,'sentiment_classifier/1/', signatures={"call": model.call})
if __name__ == '__main__':
    train()
    