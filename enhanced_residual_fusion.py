from keras.layers import LSTM, Dense, Concatenate, Dropout, MaxPooling1D, Input, Bidirectional, Conv1D, BatchNormalization, Add, GlobalMaxPooling1D, Attention, Activation, Multiply, Reshape, Flatten, Layer, Permute, MultiHeadAttention
import tensorflow as tf


class EnhancedFusionLayer(Layer):
    
    def __init__(self, num_heads, key_dim, **kwargs):
        super(EnhancedFusionLayer, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
    
    def call(self, inputs):
        concatenated_inputs = Concatenate()(inputs)
        attention_output = self.attention(concatenated_inputs, concatenated_inputs)
        return Add()([concatenated_inputs, attention_output])
    
    def get_config(self):
        config = super(EnhancedFusionLayer, self).get_config()
        config.update({
            "num_heads": self.attention.num_heads,
            "key_dim": self.attention.key_dim
        })
        return config
    
    
    