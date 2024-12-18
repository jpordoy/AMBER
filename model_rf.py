from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation,
    MaxPooling1D, Bidirectional, LSTM, Dense, Dropout,
    Reshape, Permute, Attention, Add, GlobalMaxPooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from config import Config as config



class Amber_RF:
    def __init__(self, row_hidden, col_hidden, num_classes):
        self.row_hidden = row_hidden
        self.col_hidden = col_hidden
        self.num_classes = num_classes
        self.model = None

    def conv_block(self, in_layer, filters, kernel_size):
        conv = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(in_layer)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        return conv

    def lstm_pipe(self, in_layer):
        b1 = self.conv_block(in_layer, filters=64, kernel_size=3)
        b1 = MaxPooling1D(pool_size=2)(b1)
        b2 = self.conv_block(b1, filters=128, kernel_size=3)
        b2 = MaxPooling1D(pool_size=2)(b2)
        b3 = self.conv_block(b2, filters=128, kernel_size=3)
        b3 = MaxPooling1D(pool_size=2)(b3)
        encoded_rows = Bidirectional(LSTM(self.row_hidden, return_sequences=True))(b3)
        return LSTM(self.col_hidden)(encoded_rows)

    def build_model(self, num_features, input_shape):
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
            lstm_output_reshaped = Reshape((-1, lstm_output.shape[-1]))(lstm_output)
            attention_output = Attention()([lstm_output_reshaped, lstm_output_reshaped])  # Element-wise addition
            attention_outputs.append(attention_output)

        # Combine attention outputs (equivalent to FusionLayer)
        fused_features = Add()(attention_outputs)

        # Rest of the model architecture remains the same

        dense_output = Dense(64, activation='relu', kernel_regularizer=l2(0.0001))(fused_features)
        dense_output = BatchNormalization()(dense_output)
        dense_output = Dropout(0.3)(dense_output)  # Increase dropout rate for better regularization
        dense_output = Dense(64, activation='relu', kernel_regularizer=l2(0.0001))(dense_output)
        dense_output = BatchNormalization()(dense_output)
        dense_output = Dropout(0.3)(dense_output)  # Increase dropout rate for better regularization
        prediction = Dense(self.num_classes, activation='softmax')(GlobalMaxPooling1D()(dense_output))

        model = Model(inputs=input_layers, outputs=prediction)
        self.model = model

    def compile_model(self):
        optimizer = RMSprop(learning_rate=0.000001)  # Adjust learning rate as needed
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    def train_model(self, X_train_list, y_train, X_val_list, y_val, epochs=config.epochs, batch_size=config.batch_size):
        # Define the callbacks
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.000001, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

        # Train the model with class weights and callbacks
        history = self.model.fit(
            X_train_list, y_train,
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_data=(X_val_list, y_val),
            verbose=1,
            callbacks=[reduce_lr]
        )
        return history

    def evaluate_model(self, X_test, y_test, batch_size=32):
        return self.model.evaluate(X_test, y_test, batch_size=config.batch_size)

    def predict(self, X):
        return self.model.predict(X)

    def architecture(self):
        return self.model.summary()

    def save_model(self, file_path):
        if self.model is not None:
            self.model.save(file_path)
            print(f"Model saved to {file_path}")
        else:
            raise ValueError("The model has not been built or trained yet.")