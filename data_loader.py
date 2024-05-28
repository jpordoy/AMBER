import numpy as np
import pandas as pd
from scipy import stats
from config import config

class DataLoader:
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
            label_mode = stats.mode(self.dataframe['label'][i: i + config.N_TIME_STEPS]) #Code for Python 3.7
            #label_mode = np.unique(self.dataframe['label'][i: i + config.N_TIME_STEPS])[0] #change this line for python 3.12
            if isinstance(label_mode.mode, np.ndarray):
                label = label_mode.mode[0]
            else:
                label = label_mode.mode
            segments.append(segment)
            labels.append(label)
        segments = np.asarray(segments, dtype=np.float32)
        labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)     
        return segments, labels
