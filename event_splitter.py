import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config as config

class EventSplitter:
    def __init__(self, config):
        self.config = config
    
    def format_data(self, segments, labels):
        unique_event_ids = np.unique(segments[:, :, -1])  # Get unique event IDs
        train_event_ids, test_event_ids = train_test_split(unique_event_ids, test_size=0.25, random_state=self.config.RANDOM_SEED)
        
        train_segments = segments[np.isin(segments[:, :, -1], train_event_ids)]
        test_segments = segments[np.isin(segments[:, :, -1], test_event_ids)]
        
        X_train_reshaped = self._reshape_segments(train_segments)
        X_test_reshaped = self._reshape_segments(test_segments)
        
        y_train = labels[np.isin(segments[:, :, -1], train_event_ids)]
        y_test = labels[np.isin(segments[:, :, -1], test_event_ids)]
        
        return X_train_reshaped, X_test_reshaped, y_train, y_test
