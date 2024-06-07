import numpy as np
import pandas as pd
import config as config
import tensorflow as tf
from scipy import stats
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix
from keras.optimizers import RMSprop, Adam, Adamax
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.losses import categorical_crossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, MaxPooling1D, Input, Bidirectional, Conv1D, BatchNormalization, Concatenate, Add, GlobalMaxPooling1D, Attention, Activation, Multiply, Reshape, Flatten, Layer, Permute, MultiHeadAttention
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix
from keras.regularizers import l2


# Custom DataLoader class
class CustomDataLoader:
    def __init__(self, dataframe, time_steps, step, target_column):
        self.dataframe = dataframe
        self.time_steps = time_steps
        self.step = step
        self.target_column = target_column
    
    def load_data(self):
        segments = []
        labels = []
        for i in range(0, self.dataframe.shape[0] - config.N_TIME_STEPS, config.step):  
            mag = self.dataframe['rawData'].values[i: i + config.N_TIME_STEPS]
            hr = self.dataframe['hr'].values[i: i + config.N_TIME_STEPS]
            segment = np.column_stack((mag, hr))
            label_mode = stats.mode(self.dataframe['label'][i: i + config.N_TIME_STEPS])
            if isinstance(label_mode.mode, np.ndarray):
                label = label_mode.mode[0]
            else:
                label = label_mode.mode
            segments.append(segment)
            labels.append(label)
        segments = np.asarray(segments, dtype=np.float32)
        labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)     
        return segments, labels
    

class DataFormatter:
    def __init__(self, config):
        self.config = config
    
    def format_data(self, segments, labels):
        X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.3, random_state=self.config.RANDOM_SEED)
        X_train_reshaped = self._reshape_segments(X_train)
        X_test_reshaped = self._reshape_segments(X_test)
        return X_train_reshaped, X_test_reshaped, y_train, y_test
    
    def _reshape_segments(self, segments):
        reshaped_segments = {}
        num_samples, num_time_steps, num_features = segments.shape
        for i in range(num_features):
            feature_name = f"Feature_{i+1}"
            reshaped_segments[feature_name] = segments[:, :, i].reshape(-1, num_time_steps, 1)
        return reshaped_segments

class FusionLayer(Layer):
    def __init__(self, **kwargs):
        super(FusionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return Multiply()(inputs)  # Element-wise addition

    def get_config(self):
        config = super(FusionLayer, self).get_config()
        return config
    
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
        b1 = self.conv_block(in_layer, filters=256, kernel_size=2)
        b1 = MaxPooling1D(pool_size=2, strides=2)(b1)  # Setting stride size to 2
        b2 = self.conv_block(b1, filters=128, kernel_size=2)
        b2 = MaxPooling1D(pool_size=2, strides=2)(b2)  # Setting stride size to 2
        b3 = self.conv_block(b2, filters=64, kernel_size=2)
        b3 = MaxPooling1D(pool_size=2, strides=2)(b3)  # Setting stride size to 2
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
        fused_features = EnhancedFusionLayer(num_heads=8, key_dim=32)(weighted_features)
        
        dense_output = Dense(128, activation='relu')(fused_features)
        dense_output = BatchNormalization()(dense_output)
        dense_output = Dropout(0.1)(dense_output)
        
        
        prediction = Dense(3, activation='softmax')(GlobalMaxPooling1D()(dense_output))
        model = Model(inputs=input_layers, outputs=prediction)

        self.model = model
                
        # Plot and save the model architecture
        #plot_model(self.model, to_file='plots/model_plot.png', show_shapes=True, show_layer_names=True)



    def compile_model(self):
        from keras.losses import mean_squared_error
        optimizer = RMSprop(learning_rate=0.000001)  # Set custom learning rate categorical_crossentropy
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    def train_model(self, X_train_list, y_train, X_val_list, y_val, epochs=config.epochs, batch_size=config.batch_size):
        #for X_train in X_train_list:
            #print("X_train", X_train.shape)
        #for X_val in X_val_list:
           #print("X_val", X_val.shape)
        history = self.model.fit(X_train_list, y_train, epochs=config.epochs, batch_size=config.batch_size, validation_data=(X_val_list, y_val), verbose=1)
        return history

    def evaluate_model(self, X_test, y_test, batch_size=config.batch_size):
        return self.model.evaluate(X_test, y_test, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)

    def architecture(self):
        return self.model.summary()
    
class KFoldCrossValidation:
    def __init__(self, ts_model, X_train, y_train, batch_size=config.batch_size, epochs=config.epochs, k=config.k, save_dir='plots'):
        self.ts_model = ts_model
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.epochs = epochs
        self.k = k
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)  # Create directory if it doesn't exist
        self.history_accumulated = {"accuracy": [], "loss": [], "val_accuracy": [], "val_loss": []}  # Initialize empty dictionaries to accumulate metrics
        self.fold_history = []  # Initialize list to store individual fold histories
        
    def plot_confusion_matrix(self, fold, confusion_mat):
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - Fold {fold + 1}')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig(os.path.join(self.save_dir, f'confusion_matrix_fold_{fold + 1}.png'))  # Save each plot with a unique filename
        plt.close()
        
    def plot_individual_metrics(self, fold):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Create subplots for accuracy and loss

        # Plot training accuracy
        axs[0].plot(self.fold_history[fold]['accuracy'], label='Training Accuracy')
        axs[0].plot(self.fold_history[fold]['val_accuracy'], label='Validation Accuracy')
        axs[0].set_title('Accuracy - Fold {}'.format(fold + 1))
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Accuracy')
        axs[0].legend()
        axs[0].grid(True)


        # Plot training loss
        axs[1].plot(self.fold_history[fold]['loss'], label='Training Loss')
        axs[1].plot(self.fold_history[fold]['val_loss'], label='Validation Loss')
        axs[1].set_title('Loss - Fold {}'.format(fold + 1))
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'individual_metrics_fold_{fold + 1}.png'))  # Save the plot with a unique filename
        plt.close()  # Close the plot to avoid displaying it

    def plot_overall_metrics(self):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Create subplots for accuracy, validation accuracy, loss, and validation loss

        # Plot overall accuracy
        for fold in range(self.k):
            axs[0, 0].plot(range(1, self.epochs + 1), self.fold_history[fold]['accuracy'], label=f'Fold {fold + 1}')
            axs[0, 0].set_title('Accuracy')
            axs[0, 0].set_xlabel('Epoch')
            axs[0, 0].set_ylabel('Accuracy')
            axs[0, 0].legend()
            axs[0, 0].grid(True)


        # Plot overall validation accuracy
        for fold in range(self.k):
            axs[0, 1].plot(range(1, self.epochs + 1), self.fold_history[fold]['val_accuracy'], label=f'Fold {fold + 1}')
            axs[0, 1].set_title('Validation Accuracy')
            axs[0, 1].set_xlabel('Epoch')
            axs[0, 1].set_ylabel('Accuracy')
            axs[0, 1].legend()
            axs[0, 1].grid(True)


        # Plot overall loss
        for fold in range(self.k):
            axs[1, 0].plot(range(1, self.epochs + 1), self.fold_history[fold]['loss'], label=f'Fold {fold + 1}')
            axs[1, 0].set_title('Loss')
            axs[1, 0].set_xlabel('Epoch')
            axs[1, 0].set_ylabel('Loss')
            axs[1, 0].legend()
            axs[1, 0].grid(True)


        # Plot overall validation loss
        for fold in range(self.k):
            axs[1, 1].plot(range(1, self.epochs + 1), self.fold_history[fold]['val_loss'], label=f'Fold {fold + 1}')
            axs[1, 1].set_title('Validation Loss')
            axs[1, 1].set_xlabel('Epoch')
            axs[1, 1].set_ylabel('Loss')
            axs[1, 1].legend()
            axs[1, 1].grid(True)


        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'overall_metrics.png'))
        plt.close()  # Close the plot to avoid displaying it

    def run(self):
        kf = KFold(n_splits=self.k, shuffle=True)
        all_test_losses = []
        all_test_accuracies = []
        for fold, (train_index, test_index) in enumerate(kf.split(self.X_train[0])):
            print(f"Fold {fold + 1}/{self.k}")
            X_fold_train = [X[train_index] for X in self.X_train]
            y_fold_train = self.y_train[train_index]
            X_fold_val = [X[test_index] for X in self.X_train]
            y_fold_val = self.y_train[test_index]
            self.ts_model.build_model(num_features=2, input_shape=(config.N_TIME_STEPS, 1))
            self.ts_model.compile_model()
            history = self.ts_model.train_model(X_fold_train, y_fold_train, X_fold_val, y_fold_val, epochs=self.epochs, batch_size=self.batch_size)
            test_loss, test_accuracy = self.ts_model.evaluate_model(X_fold_val, y_fold_val)
            all_test_losses.append(test_loss)
            all_test_accuracies.append(test_accuracy)
            print(f"Fold {fold + 1} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            
            # Accumulate history
            self.fold_history.append(history.history)
            
            # Generate confusion matrix
            y_pred_val = self.ts_model.predict(X_fold_val)
            y_pred_classes = np.argmax(y_pred_val, axis=1)
            y_true_classes = np.argmax(y_fold_val, axis=1)
            
            confusion_mat = confusion_matrix(y_true_classes, y_pred_classes)

            # Print confusion matrix
            print(f"Confusion Matrix for Fold {fold + 1}:\n{confusion_mat}\n")

            # Save confusion matrix plot
            self.plot_confusion_matrix(fold, confusion_mat)
        
            # Save individual plots
            self.plot_individual_metrics(fold)
        
        avg_test_loss = np.mean(all_test_losses)
        avg_test_accuracy = np.mean(all_test_accuracies)
        print(f"Average Test Loss: {avg_test_loss:.4f}, Average Test Accuracy: {avg_test_accuracy:.4f}")
        
        # Plot overall metrics
        self.plot_overall_metrics()

        return self.history_accumulated


def evaluate_model_performance(model, X_test_list, y_test_reshaped):
    # Predict classes for test data
    y_pred = model.predict(X_test_list)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test_reshaped, axis=1)

    # Calculate classification metrics
    classification_report_str = classification_report(y_true_classes, y_pred_classes, zero_division=0)
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
    cohen_kappa = cohen_kappa_score(y_true_classes, y_pred_classes)
    mcc = matthews_corrcoef(y_true_classes, y_pred_classes)
    confusion_mat = confusion_matrix(y_true_classes, y_pred_classes)

    # Calculate various metrics
    TP = np.diag(confusion_mat)
    FP = confusion_mat.sum(axis=0) - TP
    FN = confusion_mat.sum(axis=1) - TP
    TN = confusion_mat.sum() - (TP + FP + FN)

    TPR = np.divide(TP, (TP + FN), out=np.zeros_like(TP, dtype=float), where=(TP + FN)!=0)
    TNR = np.divide(TN, (TN + FP), out=np.zeros_like(TN, dtype=float), where=(TN + FP)!=0)
    PPV = np.divide(TP, (TP + FP), out=np.zeros_like(TP, dtype=float), where=(TP + FP)!=0)
    NPV = np.divide(TN, (TN + FN), out=np.zeros_like(TN, dtype=float), where=(TN + FN)!=0)
    FPR = np.divide(FP, (FP + TN), out=np.zeros_like(FP, dtype=float), where=(FP + TN)!=0)
    FNR = np.divide(FN, (TP + FN), out=np.zeros_like(FN, dtype=float), where=(TP + FN)!=0)
    FDR = np.divide(FP, (TP + FP), out=np.zeros_like(FP, dtype=float), where=(TP + FP)!=0)
    ACC = np.divide((TP + TN), (TP + FP + FN + TN), out=np.zeros_like(TP, dtype=float), where=(TP + FP + FN + TN)!=0)

    return {
        "classification_report": classification_report_str,
        "accuracy": accuracy,
        "f1": f1,
        "cohen_kappa": cohen_kappa,
        "mcc": mcc,
        "confusion_matrix": confusion_mat,
        "TPR": TPR,
        "TNR": TNR,
        "PPV": PPV,
        "NPV": NPV,
        "FPR": FPR,
        "FNR": FNR,
        "FDR": FDR,
        "ACC": ACC
    }

    

# Define your DataFrame and parameter
mypath = 'Data/Train_Me.csv'
df = pd.read_csv(mypath)
target_column = 'label'  # Name of the target column

# Step 1: Load Data
data_loader = CustomDataLoader(dataframe=df, time_steps=config.N_TIME_STEPS, step=config.step, target_column=target_column)
segments, labels = data_loader.load_data()

# Step 2: Format Data
data_formatter = DataFormatter(config=config)
X_train_reshaped, X_test_reshaped, y_train, y_test = data_formatter.format_data(segments, labels)

# Reshape y_test correctly
y_test_reshaped = np.asarray(y_test, dtype=np.float32)

# Initialize model
ts_model = PANN(row_hidden=config.row_hidden, col_hidden=config.row_hidden, num_classes=config.N_CLASSES)

# Create an instance of KFoldCrossValidation
kfold_cv = KFoldCrossValidation(ts_model, [X_train_reshaped['Feature_1'], X_train_reshaped['Feature_2']], y_train)

# Run the cross-validation
kfold_cv.run()

# Evaluate the model performance
evaluation_results = evaluate_model_performance(ts_model, [X_test_reshaped['Feature_1'], X_test_reshaped['Feature_2']], y_test_reshaped)


# Access individual metrics
print("Accuracy:", evaluation_results["accuracy"])
print("F1 Score:", evaluation_results["f1"])
print("Cohen's Kappa:", evaluation_results["cohen_kappa"])
print("MCC:", evaluation_results["mcc"])
print("Confusion Matrix:\n", evaluation_results["confusion_matrix"])
print("True Positive Rate (TPR):", evaluation_results["TPR"])
print("True Negative Rate (TNR):", evaluation_results["TNR"])
print("Positive Predictive Value (PPV):", evaluation_results["PPV"])
print("Negative Predictive Value (NPV):", evaluation_results["NPV"])
print("False Positive Rate (FPR):", evaluation_results["FPR"])
print("False Negative Rate (FNR):", evaluation_results["FNR"])
print("False Discovery Rate (FDR):", evaluation_results["FDR"])
print("Accuracy per class (ACC):", evaluation_results["ACC"])

