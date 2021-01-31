import tensorflow as tf

class DownstreamClassifier(tf.keras.Model):
    def __init__(self, layer_dims, dropout_rate, bn=True, output_unit=2):
        super().__init__()
        self.layer_dims = layer_dims
        self.dropout_rate = dropout_rate
        self.bn = bn
        self.output_unit = output_unit
        self.dropout = tf.keras.layers.Dropout(dropout_rate) if dropout_rate>0 else None
        self.denses = tf.keras.Sequential([ self._get_dense(unit, bn) for unit in layer_dims])
        self.out_layer = tf.keras.layers.Dense(output_unit, activation='softmax')
    
    def get_config(self, ):
        configs = super().get_config()
        configs.update({
            'layer_dims': self.layer_dims,
            'drouput_rate': self.dropout_rate,
            'bn': self.bn,
            'output_unit': self.output_unit
        })

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 768), dtype=tf.float32, name='input')])    
    def call(self, input):
        'for predict and serving'
        x = self.dropout(input, training=False)
        x = self.denses(x, training=False)
        out = self.out_layer(x)
        return out
    
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 768), dtype=tf.float32, name='input')])            
    def train(self, input):
        x = self.dropout(input, training=False)
        x = self.denses(x, training=True)
        out = self.out_layer(x)
        return out
    
    def _get_dense(self, unit, bn=True):
        if bn == True:
            return tf.keras.Sequential([
                tf.keras.layers.Dense(unit),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU()
            ])
        else:
            return tf.keras.Sequential([
                tf.keras.layers.Dense(unit),
                tf.keras.layers.ReLU()
            ])


class Deprecated_DownstreamClassifier(tf.keras.layers.Layer):
    'layer version, deprecated.'
    def __init__(self, layer_dims, dropout_rate, bn=True, output_unit=2):
        super().__init__()
        self.layer_dims = layer_dims
        self.dropout_rate = dropout_rate
        self.bn = bn
        self.output_unit = output_unit
        self.dropout = tf.keras.layers.Dropout(dropout_rate) if dropout_rate>0 else None
        self.denses = tf.keras.Sequential([ self._get_dense(unit, bn) for unit in layer_dims])
        self.out_layer = tf.keras.layers.Dense(output_unit, activation='softmax')
    
    def get_config(self, ):
        configs = super().get_config()
        configs.update({
            'layer_dims': self.layer_dims,
            'drouput_rate': self.dropout_rate,
            'bn': self.bn,
            'output_unit': self.output_unit
        })
        
    def call(self, inp):
        x = self.dropout(inp)
        x = self.denses(x)
        out = self.out_layer(x)
        return out
    
    def build(self, input_shape):
        super().build(input_shape)
        for sub_seq in self.denses.layers:
            for layer in sub_seq.layers:
                if isinstance(layer, tf.keras.layers.Dense):
                    layer.build(input_shape)
                    input_shape = (layer.units,)   
        #build output layer
        self.out_layer.build(input_shape)
        
        
    def _get_dense(self, unit, bn=True):
        if bn == True:
            return tf.keras.Sequential([
                tf.keras.layers.Dense(unit),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU()
            ])
        else:
            return tf.keras.Sequential([
                tf.keras.layers.Dense(unit),
                tf.keras.layers.ReLU()
            ])