import os
import tensorflow as tf

# Set the environment variable to suppress info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Now initialize TensorFlow (this will suppress info logs)
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.layers import (
    Layer, Add, Input, Conv1D, BatchNormalization, Activation,
    MaxPooling1D, Bidirectional, LSTM, Dense, Dropout,
    Reshape, Permute, Attention, GlobalMaxPooling1D, Concatenate, MultiHeadAttention
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import json
from tensorflow.keras.models import save_model as tf_save_model, load_model as tf_load_model
from tensorflow.keras.saving import register_keras_serializable

# Register the custom layer to make it serializable
@register_keras_serializable(package='custom', name='EnhancedFusionLayer')
class EnhancedFusionLayer(Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super(EnhancedFusionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads  # Store num_heads as an attribute
        self.key_dim = key_dim      # Store key_dim as an attribute
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        
    def call(self, inputs):
        # Concatenate inputs along the last axis
        concatenated_inputs = Concatenate()(inputs)
        # Apply multi-head attention to concatenated inputs
        attention_output = self.attention(concatenated_inputs, concatenated_inputs)
        # Add the original concatenated inputs to the attention output
        return Add()([concatenated_inputs, attention_output])
        
    def get_config(self):
        # Retrieve base config and update with num_heads and key_dim
        config = super(EnhancedFusionLayer, self).get_config()
        config.update({
            "num_heads": self.num_heads,  # Use stored attribute
            "key_dim": self.key_dim       # Use stored attribute
         })
        return config
