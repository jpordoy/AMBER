import pandas as pd
from sklearn.model_selection import train_test_split

class EventBasedSplitter:
    def __init__(self, dataframe, test_size=0.25, random_state=42):
        """
        Initialize the EventBasedSplitter class.
        
        :param dataframe: The input DataFrame containing the data to split.
        :param test_size: Proportion of eventIDs to include in the test set (default: 0.25).
        :param random_state: Random seed for reproducibility (default: 42).
        """
        self.df = dataframe
        self.test_size = test_size
        self.random_state = random_state

    def split_by_event(self):
        """
        Split the DataFrame into training and testing sets by eventID.
        
        :return: Two DataFrames - train_df and test_df.
        """
        # Extract unique eventIDs
        unique_event_ids = self.df['eventID'].unique()
        
        # Randomly split the eventIDs into train and test sets
        train_event_ids, test_event_ids = train_test_split(
            unique_event_ids, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        # Create training and testing datasets based on eventIDs
        train_df = self.df[self.df['eventID'].isin(train_event_ids)]
        test_df = self.df[self.df['eventID'].isin(test_event_ids)]
        
        return train_df, test_df
