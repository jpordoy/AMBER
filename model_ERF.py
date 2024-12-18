from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, MaxPooling1D, Input, Bidirectional, Conv1D, Concatenate, Permute, Reshape, Multiply, GlobalMaxPooling1D, Attention, Activation, BatchNormalization
from keras.optimizers import RMSprop
from keras.losses import categorical_crossentropy
from fusion import FusionLayer
from config import Config as config
from enhanced_residual_fusion import EnhancedFusionLayer


# PANN class (Parallel Attention Network)
class PANN:
    def __init__(self, row_hidden, col_hidden, num_classes):
        self.row_hidden = row_hidden
        self.col_hidden = col_hidden
        self.num_classes = num_classes  # Add num_classes attribute
        self.model = None


    def conv_block(self, in_layer, filters, kernel_size):
        conv = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(in_layer)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        return conv

    def lstm_pipe(self, in_layer):
        b1 = self.conv_block(in_layer, filters=256, kernel_size=3)
        b1 = MaxPooling1D(pool_size=3)(b1)
        b2 = self.conv_block(b1, filters=128, kernel_size=3)
        b2 = MaxPooling1D(pool_size=3)(b2)
        b3 = self.conv_block(b2, filters=64, kernel_size=3)
        b3 = MaxPooling1D(pool_size=3)(b3)
        encoded_rows = Bidirectional(LSTM(self.row_hidden, return_sequences=True))(b3)
        return LSTM(self.col_hidden)(encoded_rows)

    def build_model(self, num_features, input_shape, feature_names=None):
        input_layers = []
        lstm_outputs = []
        for i in range(num_features):
            input_layer = Input(shape=input_shape, name=f'input_feature_{i+1}')
            input_layers.append(input_layer)
            lstm_output = self.lstm_pipe(Permute(dims=(1, 2))(input_layer))
            lstm_output_reshaped = Reshape((-1,))(lstm_output)  
            lstm_outputs.append(lstm_output_reshaped)  
        
        attention_outputs = []
        for i, lstm_output in enumerate(lstm_outputs):
            # Print the shape of the LSTM output before reshaping
            #print(f"Shape of LSTM output {i + 1} before reshaping: {lstm_output.shape}")

            # Reshape the LSTM output to make it 2D
            lstm_output_reshaped = Reshape((-1, lstm_output.shape[-1]))(lstm_output)

            # Print the shape of the reshaped LSTM output
            #print(f"Shape of LSTM output {i + 1} after reshaping: {lstm_output_reshaped.shape}")

            # Pass the reshaped LSTM output to the Attention layer
            attention_output = Attention()([lstm_output_reshaped, lstm_output_reshaped])
            attention_outputs.append(attention_output)
        
        #https://docs.python.org/2/library/functions.html#zip
        # use zip for element wise addition of LSTM and ATTENTION outputs
        
        weighted_features = []
        for i, (lstm_output, attention_output) in enumerate(zip(lstm_outputs, attention_outputs)):
            weighted_feature = Multiply()([lstm_output, attention_output])
            weighted_features.append(weighted_feature)
            
        #concatenated_features = Concatenate()(weighted_features)
            #fused_features = FusionLayer()(weighted_features)
        fused_features = EnhancedFusionLayer(num_heads=4, key_dim=32)(weighted_features)
        
        dense_output = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(fused_features)
        dense_output = BatchNormalization()(dense_output)
        dense_output = Dropout(0.5)(dense_output)
        
        prediction = Dense(3, activation='softmax')(GlobalMaxPooling1D()(dense_output))
        model = Model(inputs=input_layers, outputs=prediction)

        self.model = model
                
        # Plot and save the model architecture
        #plot_model(self.model, to_file='plots/model_plot.png', show_shapes=True, show_layer_names=True)


    def compile_model(self):
        self.model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])


    def train_model(self, X_train_list, y_train, X_val_list, y_val, epochs=25, batch_size=64):
        #for X_train in X_train_list:
            #print("X_train", X_train.shape)
        #for X_val in X_val_list:
           #print("X_val", X_val.shape)
        history = self.model.fit(X_train_list, y_train, epochs=25, batch_size=32, validation_data=(X_val_list, y_val), verbose=1)
        return history

    def evaluate_model(self, X_test, y_test, batch_size=25):
        return self.model.evaluate(X_test, y_test, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)

    def architecture(self):
        return self.model.summary()
    
    #C:\Users\jamie\Documents\Python Scripts\ParallelAttentionNetwork\plots RMSprop_MSE\individual_metrics_fold_3.png
