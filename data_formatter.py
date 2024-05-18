import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import config

class DataFormatter:
    def __init__(self, config):
        self.config = config
    
    def format_data(self, segments, labels):
        X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.25, random_state=self.config.RANDOM_SEED)
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
