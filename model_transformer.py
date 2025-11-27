import tensorflow as tf

def transformer_encoder(x, num_heads=2, key_dim=32):
    attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = tf.keras.layers.Add()([x, attn])
    x = tf.keras.layers.LayerNormalization()(x)
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(x.shape[-1])
    ])
    x2 = ffn(x)
    return tf.keras.layers.Add()([x, x2])

def build_transformer(window):
    inp = tf.keras.Input(shape=(window,1))
    x = tf.keras.layers.Dense(32)(inp)
    x = transformer_encoder(x)
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(1)(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer='adam', loss='mse')
    return m
