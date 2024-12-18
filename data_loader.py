import numpy as np
import pandas as pd
from scipy import stats
from config import Config as config

class DataLoader:
    def __init__(self, dataframe, time_steps, step, target_column):
        self.dataframe = dataframe
        self.time_steps = time_steps
        self.step = step
        self.target_column = target_column

    def load_data(self):
        segments = []
        labels = []
        event_ids = []
        user_ids = []

        # Group data by eventID to ensure events are kept intact
        grouped = self.dataframe.groupby('eventID')

        for event_id, group in grouped:
            if len(group) >= self.time_steps:  # Process if the event group has enough data
                for i in range(0, len(group) - self.time_steps + 1, self.step):
                    mag = group['rawData'].values[i: i + self.time_steps]
                    hr = group['ppg'].values[i: i + self.time_steps]
                    segment = np.column_stack((mag, hr))  # Combine magnitude and heart rate features
                    label_mode = stats.mode(group[self.target_column][i: i + self.time_steps])
                    if isinstance(label_mode.mode, np.ndarray):
                        label = label_mode.mode[0]
                    else:
                        label = label_mode.mode

                    segments.append(segment)
                    labels.append(label)
                    event_ids.append(event_id)
                    user_ids.append(group['userID'].iloc[0])  # Assuming userID is consistent within an event

        # Convert to numpy arrays
        segments = np.asarray(segments, dtype=np.float32)
        labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)

        # Create DataFrame to store eventID and userID alongside segments and labels
        df_labels = pd.DataFrame({
            'segments': list(segments),
            'labels': list(labels),
            'eventID': event_ids,
            'userID': user_ids
        })

        return df_labels