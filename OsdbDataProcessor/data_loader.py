import numpy as np
import pandas as pd
from scipy import stats
from config import Config

class DataLoader:
    def __init__(self, dataframe, time_steps, step, target_column, feature_columns):
        """
        Initializes the DataLoader with the necessary parameters.
        
        :param dataframe: The input dataframe containing the dataset.
        :param time_steps: Number of time steps to include in each segment.
        :param step: The step size for segmenting the data.
        :param target_column: The target column for labeling.
        :param feature_columns: List of feature columns to include in each segment.
        """
        self.dataframe = dataframe
        self.time_steps = time_steps
        self.step = step
        self.target_column = target_column
        self.feature_columns = feature_columns  # List of feature columns to be used

    def load_data(self):
        """
        Loads and processes the data, extracting the required features for model training.
        
        :return: A DataFrame containing the segments, labels, and associated event and user IDs.
        """
        segments = []
        labels = []
        event_ids = []
        user_ids = []

        # Group data by eventId to ensure events are kept intact
        grouped = self.dataframe.groupby('eventId')

        for event_id, group in grouped:
            if len(group) >= self.time_steps:  # Process if the event group has enough data
                for i in range(0, len(group) - self.time_steps + 1, self.step):
                    # Dynamically extract the features based on the provided columns
                    segment = []
                    for feature in self.feature_columns:
                        segment.append(group[feature].values[i: i + self.time_steps])
                    
                    segment = np.column_stack(segment)  # Stack the selected features into a single segment
                    
                    # Calculate the most frequent label (mode) for the current time step segment
                    label_mode = stats.mode(group[self.target_column][i: i + self.time_steps], keepdims=True)
                    
                    if isinstance(label_mode.mode, np.ndarray):
                        label = label_mode.mode[0]
                    else:
                        label = label_mode.mode

                    segments.append(segment)
                    labels.append(label)
                    event_ids.append(event_id)
                    user_ids.append(group['userId'].iloc[0])  # Assuming userID is consistent within an event

        # Convert to numpy arrays
        segments = np.asarray(segments, dtype=np.float32)
        labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)

        # Create DataFrame to store eventID and userID alongside segments and labels
        df_labels = pd.DataFrame({
            'segments': list(segments),
            'label': list(labels),
            'eventId': event_ids,
            'userId': user_ids
        })

        return df_labels
