import json
import numpy as np
import pandas as pd

class OsdbDataLoader:
    def __init__(self, file_path, time_steps):
        self.file_path = file_path
        self.time_steps = time_steps
        self.df_sensordata = None
        self.load_and_process_data_from_json()

    def load_and_process_data_from_json(self):
        """
        Load and process OSDB data from a JSON file. This function will create a DataFrame 
        with the necessary columns and calculate FFT features.
        """
        with open(self.file_path, 'r') as file:
            raw_json = json.load(file)

        # Flatten the JSON and extract the necessary data
        flattened_data = []
        for attribute in raw_json:
            user_id = attribute.get('userId', None)
            datapoints = attribute.get('datapoints', [])

            for point in datapoints:
                event_id = point.get('eventId', None)
                hr = point.get('hr', None)
                o2Sat = point.get('o2Sat', None)
                rawData = point.get('rawData', [])
                rawData3D = point.get('rawData3D', [])

                # FFT calculation for rawData
                fft_result = self.calculate_fft(rawData)

                flattened_data.append({
                    'eventId': event_id,
                    'userId': user_id,
                    'hr': hr,
                    'o2Sat': o2Sat,
                    'rawData': rawData,
                    'rawData3D': rawData3D,
                    'FFT': fft_result  # Adding FFT column directly
                })

        # Create DataFrame from the flattened data
        self.df_sensordata = pd.DataFrame(flattened_data)

    def calculate_fft(self, raw_data):
        if not raw_data:
            return []

        # Convert raw_data to numpy array
        raw_data = np.array(raw_data)
        # Perform FFT, remove DC component, and return magnitudes
        raw_data = raw_data - np.mean(raw_data)
        fft_result = np.fft.fft(raw_data)
        fft_magnitude = np.abs(fft_result)
        # Isolate positive frequencies
        positive_fft_magnitude = fft_magnitude[:len(fft_magnitude) // 2]
        
        return positive_fft_magnitude.tolist()  # Return as a list


