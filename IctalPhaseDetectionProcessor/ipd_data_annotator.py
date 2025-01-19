import pandas as pd

class DataAnnotator:
    def __init__(self, data_file, ids):
        # Initialize the class with the data file and list of event IDs
        self.data_file = data_file
        self.ids = ids
        self.df_labels = None
        self.df_sensor_data_filtered = None
        self.df_sensor_data = None
        
        # Load the data from the provided file
        self.load_data()

    def load_data(self):
        """Load data from CSV and prepare the label DataFrame."""
        # Read the CSV file
        df = pd.read_csv(self.data_file)
        
        # Select only the required columns
        self.df_labels = df[["Id", "eventId", "label"]]
        
        # Sort the labels based on 'Id'
        self.df_labels = self.df_labels.sort_values(by='Id').reset_index(drop=True)

    def filter_and_sort_sensor_data(self, interpolated_df):
        """Filter sensor data based on eventIds and sort the DataFrame."""
        # Filter the sensor data based on eventIds
        self.df_sensor_data_filtered = interpolated_df[interpolated_df['eventId'].isin(self.ids)]
        
        # Define 'eventId' as a categorical column with the desired order
        self.df_sensor_data_filtered['eventId'] = pd.Categorical(self.df_sensor_data_filtered['eventId'], categories=self.ids, ordered=True)
        
        # Sort the sensor data by 'eventId'
        self.df_sensor_data_filtered = self.df_sensor_data_filtered.sort_values('eventId')

    def merge_labels(self):
        """Merge the label column from df_labels to df_sensor_data_filtered based on eventId."""
        # Drop duplicates to ensure unique eventIds
        df_labels_unique = self.df_labels.drop_duplicates(subset='eventId', keep='first')
        
        # Merge the labels with the filtered sensor data
        self.df_sensor_data = pd.merge(self.df_sensor_data_filtered, df_labels_unique[['eventId', 'label']], on='eventId', how='left')

    def get_sensor_data(self):
        """Return the processed sensor data."""
        return self.df_sensor_data
