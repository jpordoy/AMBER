import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config as config

class DataFormatter:
    def __init__(self, config):
        self.config = config

    def format_data(self, df_labels, priority_test_event_ids):
        # Separate priority test event IDs
        test_event_ids = set(priority_test_event_ids)
        unique_event_ids = df_labels['eventID'].unique()
        remaining_event_ids = set(unique_event_ids) - test_event_ids

        # Assign remaining events to the training set
        train_event_ids = list(remaining_event_ids)

        # Filter DataFrame for training and testing based on eventID
        df_train = df_labels[df_labels['eventID'].isin(train_event_ids)]
        df_test = df_labels[df_labels['eventID'].isin(test_event_ids)]

        # Extract segments and labels for training and testing sets
        X_train = np.asarray(df_train['segments'].tolist(), dtype=np.float32)
        y_train = np.asarray(df_train['labels'].tolist(), dtype=np.float32)
        X_test = np.asarray(df_test['segments'].tolist(), dtype=np.float32)
        y_test = np.asarray(df_test['labels'].tolist(), dtype=np.float32)

        # Reshape the segments to include feature dimension
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