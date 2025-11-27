import tensorflow as tf

class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    def call(self, query, values):
        scores = tf.matmul(query, values, transpose_b=True)
        weights = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(weights, values)

def build_model(window):
    inputs = tf.keras.Input(shape=(window,1))
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    q = tf.keras.layers.Dense(64)(x)
    att = Attention()(q, x)
    x = tf.keras.layers.Flatten()(att)
    x = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, x)
    model.compile(optimizer='adam', loss='mse')
    return model
