import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

class IpdInterpolator:
    def __init__(self, df, column_to_interpolate):
        """
        Initialize the Interpolator class with a DataFrame and the column to interpolate.
        """
        self.df = df
        self.column_to_interpolate = column_to_interpolate

    def interpolate_column(self, new_column_name='interpolated_hr', interval=125, time_step=5):
        """
        Interpolate the specified column using the provided logic.
        
        Parameters:
        - new_column_name: Name of the new column to store interpolated values.
        - interval: Interval to sample the original column (e.g., every 125th element).
        - time_step: Time step in seconds for the interpolation process.
        """
        try:
            # Step 1: Extract every nth element from the specified column
            original_values = self.df[self.column_to_interpolate]
            selected_elements = original_values[0::interval]
            x = np.array(selected_elements)

            # Step 2: Create an array representing the time (in `time_step` intervals)
            time_values = np.arange(len(x)) * time_step

            # Step 3: Create a CubicSpline object for interpolation
            cs = CubicSpline(time_values, x, bc_type='clamped')

            # Step 4: Generate new time values for finer granularity
            num_original_points = len(x)
            new_time_values = np.linspace(0, (num_original_points - 1) * time_step, num_original_points * interval)

            # Step 5: Generate interpolated values
            interpolated_values = cs(new_time_values)

            # Step 6: Update the DataFrame with the interpolated values
            self.df[new_column_name] = interpolated_values[:len(self.df)]  # Match the original DataFrame length

            print(f"Interpolation completed. New column '{new_column_name}' added to the DataFrame.")
        except Exception as e:
            print("An error occurred during interpolation:", e)

    def get_dataframe(self):
        """
        Return the updated DataFrame with interpolated values.
        """
        return self.df