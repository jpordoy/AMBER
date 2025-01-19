import pandas as pd
import numpy as np

class IpdDataReshaper:
    def __init__(self, dataframe):
        self.df = dataframe

    def reshape_data(self):
        reshaped_rows = []
        
        for idx, row in self.df.iterrows():
            event_id = row['eventId']
            user_id = row['userId']
            hr = row['hr']
            o2Sat = row['o2Sat']
            rawData = row['rawData']
            rawData3D = row['rawData3D']
            fft = row['FFT']
            
            # Replicate eventId, userId, hr, o2Sat for 125 times
            repeated_info = {
                'eventId': [event_id] * 125,
                'userId': [user_id] * 125,
                'hr': [hr] * 125,
                'o2Sat': [o2Sat] * 125
            }
            
            # Transpose rawData and FFT
            rawData_transposed = rawData[:125]  # Transpose to the correct shape
            fft_transposed = fft[:125]  # Transpose to the correct shape
            
            # Process rawData3D if it exists
            if rawData3D:
                # Convert rawData3D into lists of 3 (x, y, z)
                rawData3D_transposed = [rawData3D[i:i+3] for i in range(0, len(rawData3D), 3)]
                rawData3D_transposed = rawData3D_transposed[:125]  # Ensure only 125 rows
            else:
                rawData3D_transposed = [None] * 125  # If no rawData3D, set it to None
            
            # Create the reshaped row
            for i in range(125):
                reshaped_rows.append({
                    'eventId': repeated_info['eventId'][i],
                    'userId': repeated_info['userId'][i],
                    'hr': repeated_info['hr'][i],
                    'o2Sat': repeated_info['o2Sat'][i],
                    'rawData': rawData_transposed[i],
                    'rawData3D': rawData3D_transposed[i],
                    'FFT': fft_transposed[i]
                })
        
        # Create a new DataFrame from the reshaped rows
        reshaped_df = pd.DataFrame(reshaped_rows)
        return reshaped_df